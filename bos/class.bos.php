<?php
date_default_timezone_set("America/New_York");
require_once 'db.php';
require_once 'credentials.php';
error_reporting(-1);
ini_set('display_errors', 'On');
/**
 * Roughly https://github.com/EnvironmentalDashboard/includes/blob/master/class.BuildingOS.php but adapted for our needs
 */
class BuildingOS {
  
  /**
   * @param $db The database connection
   *
   * Sets the token for the class.
   */
  public function __construct($db) {
    $this->db = $db;
    $results = $db->query("SELECT token, token_updated FROM api LIMIT 1"); // api is a table with 1 row
    $arr = $results->fetch();
    if ($arr['token_updated'] + 3595 > time()) { // 3595 = 1 hour - 5 seconds to be safe (according to API docs, token expires after 1 hour)
      $this->token = $arr['token'];
    }
    else { // amortized cost
      $url = 'https://api.buildingos.com/o/token/';
      $data = array(
        'client_id' => CLIENT_ID,
        'client_secret' => CLIENT_SECRET,
        'username' => USERNAME,
        'password' => PASSWORD,
        'grant_type' => 'password'
        );
      $options = array(
        'http' => array(
          'method'  => 'POST',
          'header' => "Content-Type: application/x-www-form-urlencoded\r\n",
          'content' => http_build_query($data)
          )
      );
      $context = stream_context_create($options);
      $result = file_get_contents($url, false, $context);
      if ($result === false) {
        // Should handle errors better
        die("There was an error connecting with Lucid's servers.\n\n");
      }
      $json = json_decode($result, true);
      $this->token = $json['access_token'];
      $stmt = $db->prepare('UPDATE api SET token = ?, token_updated = ?');
      $stmt->execute(array($this->token, time()));
    }
  }


  /*
    ====== METHODS TO RETRIEVE DATA FROM THE API ======
   */
  

  /**
   * Makes a call to the given URL with the 'Authorization: Bearer $token' header.
   *
   * @param $url to fetch
   * @param $debug if set to true will output the URL used
   * @return contents of web page or false if there was an error
   */
  private function makeCall($url, $debug = false) {
    if ($debug) {
      echo "URL: {$url}\n\n";
    }
    $options = array(
      'http' => array(
        'method' => 'GET',
        'header' => 'Authorization: Bearer ' . $this->token
        )
    );
    $context = stream_context_create($options);
    $data = file_get_contents($url, false, $context);
    if ($data === false) { // If the API didnt return a proper response
      if ($debug) {
        print_r($http_response_header);
      }
      if ($http_response_header[0] === 'HTTP/1.1 429 TOO MANY REQUESTS') {
        // If it was because the API is being queried too quickly, sleep
        sleep( 1 + preg_replace('/\D/', '', $http_response_header[5]) );
      }
      // Try again
      $data = file_get_contents($url, false, $context);
    } 
    return $data;
  }

  /**
   * Fetches data for a meter.
   *
   * @param $meter url e.g. https://api.buildingos.com/meters/oberlin_harkness_main_e/data
   * @param $res can be day, hour, or live
   * @param $start start unix timestamp
   * @param $end end unix timestamp
   * @param $debug if set to true will output the URL used
   * @return contents of web page or false if there was an error
   */
  public function getMeter($meter, $res, $start, $end, $debug = false) {
    $start = date('c', $start);
    $end = date('c', $end);
    if ($start === false || $end === false) {
      die('Error parsing $start/$end dates');
    }
    $res = strtolower($res);
    if ($res != "live" && $res != "hour" && $res != "quarterhour" && $res != "day" && $res != "month") {
      die('$res must be live/quarterhour/hour/day/month');
    }
    $data = array(
      'resolution' => $res,
      'start' => $start,
      'end' => $end
    );
    $data = http_build_query($data);
    return $this->makeCall($meter . "?" . $data, $debug);
  }

  /**
   * Retrieves a list of buildings with their meter and other data stored in a multidimensional array.
   * @param $org array of organization URLs to restrict data collection to. If empty, buildings for all orgs will be collected
   */
  public function getBuildings($org = array()) {
    $url = 'https://api.buildingos.com/buildings?per_page=100';
    $buffer = array();
    $i = 0;
    $j = 0;
    $not_empty = !empty($org);
    while (true) {
      $result = $this->makeCall($url);
      if ($result === false) {
        throw new Exception('Failed to open building URL ' . $url);
        return false;
      }
      $json = json_decode($result, true);
      foreach ($json['data'] as $building) {
        if ($not_empty && !in_array($building['organization'], $org)) {
          continue;
        }
        echo "Fetched building: {$building['name']}\n";
        $org_id = $this->orgURLtoID($building['organization']);
        $buffer[$i] = array( // make an array that can be fed directly into a query 
          ':id' => $building['id'],
          ':org_id' => $org_id,
          ':name' => $building['name'],
          ':type' => $building['buildingType']['displayName'],
          ':address' => "{$building['address']} {$building['postalCode']}",
          ':loc' => "{$building['location']['lat']},{$building['location']['lon']}",
          ':area' => ($building['area'] == '') ? 0 : $building['area'],
          ':occupancy' => $building['occupancy'],
          ':numFloors' => $building['numFloors'],
          'meters' => array() // remove if feeding directly into query
        );
        foreach ($building['meters'] as $meter) {
          $meter_result = $this->makeCall($meter['url']);
          if ($result === false) {
            throw new Exception('Failed to open meter URL ' . $meter['url']);
            return false;
          }
          $meter_json = json_decode($meter_result, true);
          $arr = array(
            ':uuid' => $meter_json['data']['uuid'],
            ':building_id' => null, // need to fill this in later if inserting into db
            ':scope' => $meter_json['data']['scope']['displayName'],
            ':resource' => $meter_json['data']['resourceType']['displayName'],
            ':name' => $meter_json['data']['displayName'],
            ':url' => $meter_json['data']['url'],
            ':units' => $meter_json['data']['displayUnit']['displayName']
          );
          $buffer[$i]['meters'][$j] = $arr;
          $j++;
        }
        $i++;
      }
      if ($json['links']['next'] == "") { // No other data
        return $buffer;
      }
      else { // Other data to fetch
        $url = $json['links']['next'];
      }
    }
  }

  /**
   * Returns all the organizations for a buildingos account.
   * Can feed directly into $this->getBuildings($this->getOrganizations())
   */
  public function getOrganizations() {
    $buffer = array();
    $json = json_decode($this->makeCall('https://api.buildingos.com/organizations'), true);
    if ($json === false) {
      return false;
    }
    foreach ($json['data'] as $organization) {
      $buffer[$organization['name']] = $organization['url'];
    }
    return $buffer;
  }

  private function orgURLtoID($url) {
    $result = explode('/', $url);
    return intval($result[count($result)-1]);
  }


  /*
    ====== METHODS TO UPDATE THE DATABASE WITH BUILDING/METER DATA ======
   */

  /**
   * Update individual meters
   * @param   $meter_id  
   * @param   $meter_url 
   * @param   $res       
   * @param   $chunk     max amount of data to request at one time
   * @return [type]            [description]
   */
  public function updateMeter($meter_id, $meter_url, $res, $chunk) {
    $time = time(); // end date
    // Get the most recent recording. Data fetched from the API will start at $last_recording and end at $time
    $stmt = $this->db->prepare('SELECT recorded FROM meter_data
      WHERE meter_id = ? AND resolution = ? ORDER BY recorded DESC LIMIT 1');
    $stmt->execute(array($meter_id, $res));
    $last_recording = ($stmt->rowCount() === 1) ? $stmt->fetchColumn() : strtotime('-2 years'); // start date
    $diff = $time - $last_recording;
    if ($diff > $chunk) {
      $meter_data = array();
      $start = $last_recording;
      $end = $start + $chunk;
      while (true) {
        $tmp = $this->getMeter($meter_url, $res, $start, $end);
        if ($tmp === false) {
          echo "Error fetching data for meter {$meter_id} from {$start} to {$end}\n";
        }
        $tmp = json_decode($tmp, true)['data'];
        $meter_data = array_merge($meter_data, $tmp);
        $start = $end;
        $end += $chunk;
        if ($end >= $time) {
          break;
        }
      }
    } else {
      $meter_data = $this->getMeter($meter_url, $res, $last_recording, $time);
      if ($meter_data === false) { // file_get_contents returned false, so problem with API
        return false;
      }
      $meter_data = json_decode($meter_data, true);
      $meter_data = $meter_data['data'];
    }
    if (!empty($meter_data)) {
      $last_value = null;
      $last_recorded = null;
      foreach ($meter_data as $data) { // Insert new data
        $localtime = strtotime($data['localtime']);
        if ($localtime > $last_recording) { // just to make sure
          $stmt = $this->db->prepare('INSERT INTO meter_data (meter_id, value, recorded, resolution) VALUES (?, ?, ?, ?)');
          $stmt->execute(array($meter_id, $data['value'], $localtime, $res));
          if ($data['value'] !== null) {
            $last_value = $data['value'];
            $last_recorded = $localtime;
          }
        }
      }
    } // if !empty($meter_data)
    return true;
  }

  /**
   * Adds buildings from the BuildingOS API that aren't already in the database.
   * Optionally delete buildings/meters that no longer exist in the API
   * @param  $org array fed into getBuildings()
   * @param  $delete_not_found delete buildings/meters that exist in the database but not the API
   */
  public function syncBuildings($org, $delete_not_found = false) {
    // Get a list of all buildings to compare against what's in db
    $buildings = $this->getBuildings($org);
    echo "Fetched all buildings\n";
    if ($buildings !== false) {
      if ($delete_not_found) { // Delete buildings in db not found in $buildings
        $bos_ids = array_column($buildings, ':id');
        foreach ($this->db->query("SELECT id FROM buildings") as $building) {
          if (!in_array($building['id'], $bos_ids)) {
            $stmt = $this->db->prepare('DELETE FROM buildings WHERE id = ?');
            $stmt->execute(array($building['id']));
            // also delete meters that belong to those buildings
            $stmt = $this->db->prepare('DELETE FROM meters WHERE building_id = ?');
            $stmt->execute(array($building['id']));
            // also delete data from meter_data table
            $stmt = $this->db->prepare('DELETE FROM meter_data WHERE meter_id IN (SELECT id FROM meters WHERE building_id = ?)');
            $stmt->execute(array($building['id']));
          }
        }
      }
      $counter = 0;
      foreach ($buildings as $building) {
        echo "Processing building " . (++$counter) . " out of " . count($buildings) . "\n";
        $building_id = $building[':id'];
        $stmt = $this->db->prepare('SELECT COUNT(*) FROM buildings WHERE id = ?');
        $stmt->execute(array($building_id));
        if ($stmt->fetchColumn() === '0') { // building doesnt exist in db
          $stmt = $this->db->prepare('INSERT INTO buildings (id, org_id, name, type, address, loc, area, occupancy, floors) VALUES (:id, :org_id, :name, :type, :address, :loc, :area, :occupancy, :numFloors)');
          foreach (array(':id', ':org_id', ':name', ':type', ':address', ':loc', ':area', ':occupancy', ':numFloors') as $param) {
            $stmt->bindValue($param, $building[$param]);
          }
          $stmt->execute();
        }
        // $building is now guaranteed to be a row in the db
        if ($delete_not_found) { // delete meters not found in $buildings['meters']
          $bos_uuids = array_column($building['meters'], ':uuid');
          foreach ($this->db->query('SELECT id, uuid FROM meters WHERE building_id = ' . intval($building_id)) as $meter) {
            if (!in_array($meter['uuid'], $bos_uuids)) {
              $stmt = $this->db->prepare('DELETE FROM meters WHERE uuid = ?');
              $stmt->execute(array($meter['uuid']));
              $stmt = $this->db->prepare('DELETE FROM relative_values WHERE meter_uuid = ?');
              $stmt->execute(array($meter['uuid']));
              $stmt = $this->db->prepare('DELETE FROM meter_data WHERE meter_id = ?');
              $stmt->execute(array($meter['id']));
            }
          }
        }
        // make sure all the meters are there
        foreach ($building['meters'] as $meter) {
          $stmt = $this->db->prepare('SELECT COUNT(*) FROM meters WHERE url = ?');
          $stmt->execute(array($meter[':url']));
          if ($stmt->fetchColumn() === '0') { // meter is not in db
            $meter[':building_id'] = $building_id;
            $stmt = $this->db->prepare('INSERT INTO meters (uuid, building_id, scope, resource, name, url, units) VALUES (:uuid, :building_id, :scope, :resource, :name, :url, :units)');
            $stmt->execute($meter);
          }
        }
      }
    }
  }
  
}
?>