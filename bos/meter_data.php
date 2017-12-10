<?php
require_once 'class.bos.php';
ignore_user_abort(1);
set_time_limit(0);
$bos = new BuildingOS($db);
$res = 'hour';
$chunk = 604800; // download data in week chunks
foreach ($db->query('SELECT id, url FROM meters') as $meter) {
	$bos->updateMeter($meter['id'], $meter['url'] . '/data', $res, $chunk);
	sleep(3); // don't bombard bos api
}
?>