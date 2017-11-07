<?php
require_once 'class.bos.php';
$bos = new BuildingOS($db);
$res = 'hour';
$chunk = 86400;
foreach ($db->query('SELECT id, url FROM meters') as $meter) {
	$bos->updateMeter($meter['id'], $meter['url'], $res, $chunk);
	sleep(3); // don't bombard bos api
}
?>