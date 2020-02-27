<?php
	$output = shell_exec("python -W ignore web_chi_lo_ha_detto.py 2>&1");
	 
	$delimiter =  "]"; 
	$tokens = array();
	$n_tokens = 0;

	$token = strtok($output, $delimiter); 
	$tokens[$n_tokens] = substr($token, 2);
	$n_tokens++;
	   
	while($n_tokens < 3)   
	{   
		//echo $tokens[$n_tokens-1]."\n";
	    $token = strtok($delimiter); 
	    $tokens[$n_tokens] = substr($token, 3); 
	    $n_tokens++;
	}

	$tokens[$n_tokens-1] = substr($tokens[$n_tokens-1], 1, strlen($tokens[$n_tokens-1])-2);
?>

<html>
	<head><title>Classificatore di messaggi</title></head>
	<body>
		<?php echo $tokens[$n_tokens-1] ?>		
	</body>
</html>