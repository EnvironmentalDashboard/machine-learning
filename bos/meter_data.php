<?php
require_once 'class.bos.php';
ignore_user_abort(1);
set_time_limit(0);
$bos = new BuildingOS($db);
$res = 'day';
$chunk = 8640000; // download data in 100 day chunks
foreach ($db->query('SELECT id, url FROM meters') as $meter) {
	$bos->updateMeter($meter['id'], $meter['url'] . '/data', $res, $chunk);
	sleep(3); // don't bombard bos api
}
?>