GPU=$1

## textual inversion
bash scripts/textual_inversion/textual_inversion_sd.sh $GPU space-shuttle _shuttle_ shuttle

## static stage
bash scripts/animate124/run-static.sh $GPU run space-shuttle "A high-resolution DSLR image of a space shuttle launching"

## dynamic corase stage
bash scripts/animate124/run-dynamic.sh  $GPU run run space-shuttle "a space shuttle launching"

## dynamic fine stage
bash scripts/animate124/run-cn.sh $GPU run run run space-shuttle "a space shuttle launching" "a <token> launching"