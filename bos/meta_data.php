<?php
require_once 'class.bos.php';
$bos = new BuildingOS($db);
// To get the organization URLs fed into syncBuildings():
// foreach ($bos->getOrganizations() as $org_name => $org_url)
// 	echo "$org_name $org_url\n";
$bos->syncBuildings(array('https://api.buildingos.com/organizations/112', 'https://api.buildingos.com/organizations/864', 'https://api.buildingos.com/organizations/500'));
// 'Oberlin College', 'City of Oberlin, Private Organizations', and 'City of Oberlin'
?>