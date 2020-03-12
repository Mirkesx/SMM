<?php
	if($_SERVER['REQUEST_METHOD'] == 'POST') {
		$message = $_POST["mess"];
		$message = str_replace('"',' ',$message);
		$message = str_replace("'"," ",$message);

		$type = $_POST["type"];




		$output = shell_exec("/home/mc/anaconda3/bin/python3.7 -W ignore web_chi_lo_ha_detto_load_model.py '".$message."' '".$type."' 'LOG_REG' 2>&1");
		 
		$delimiter =  "]"; 
		$tokens = array();
		$n_tokens = 0;

		$token = strtok($output, $delimiter); 
		$tokens[$n_tokens] = substr($token, 2);
		$n_tokens++;
		   
		while($n_tokens < 5)   
		{   
		    $token = strtok($delimiter); 
		    $tokens[$n_tokens] = substr($token, 2); 
		    $n_tokens++;
		}

		$persona = substr($tokens[0], 1, strlen($tokens[0])-2);
		$prob = substr($tokens[1], 7, strlen($tokens[1])-3);
		$table_weights = substr($tokens[2], 3, strlen($tokens[2])-4);
		$table_explainations = substr($tokens[3], 2, strlen($tokens[3])-3);
	}
?>

<html>
	<head><title>Classificatore di messaggi</title></head>
	<body>
		<?php
			if($_SERVER['REQUEST_METHOD'] == 'POST') {
				if($type == 'POLITICO')
					echo "<h1>Quale politico sei?</h1>";
				else
					echo "<h1>Quale personaggio famoso sei?</h1>";

				echo "<img src='../WEB-DATA/FOTO/".$persona.".jpg'>";

				echo "<br><br><b>".$persona."</b> con una probabilità del ".($prob*100)."%.<br><br>";
		?>

			<form method='GET' action='<?= htmlspecialchars($_SERVER['PHP_SELF']) ?>' >
		        <br><input type='submit' value='Ritorna al prompt'>
		    </form>

		<?php

				echo "<h2>Tabella dei pesi delle features</h2>";

				echo "Questa tabella spiega come il classificatore pesa ogni parola o gruppo di parole.<br><br>";

				echo $table_weights."<br>";


				echo "<h2>Features utilizzate nella scelta</h2>Questo gruppo di tabelle indicano invece le probabilità e i punteggi che il classificatore ha dato per ogni etichetta.<br>Se posizioni il mouse sopra alle parole colorate potrai vedere di quanto hanno migliorato o peggiorato il punteggio.<br><br><br>";

				echo $table_explainations."<br><br>";
			}
			else {
		?>
				<h1>Classificatore di messaggi</h1>
				<form method='POST' action='<?= htmlspecialchars($_SERVER['PHP_SELF']) ?>' >
		            <br>Messaggio:
		            <br><textarea name="mess" rows="4" cols="50"></textarea>
		            <br>Politici italiani<input type="radio" name="type" value="POLITICO" checked>
		            <br>Personaggi dello spettacolo italiani<input type="radio" name="type" value="CONDUTTORE">
		            <br><input type='submit' value='Invia'>
		        </form>
		<?php
			}
		?>
			
	</body>
</html>