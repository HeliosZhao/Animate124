GPU=$1

## textual inversion
bash scripts/textual_inversion/textual_inversion_sd.sh $GPU astronaut-horse _astronaut_ astronaut

## static stage
bash scripts/animate124/run-static.sh $GPU run astronaut-horse "A high-resolution DSLR image of an astronaut, helmet in hand, rides a white horse"

## dynamic corase stage
bash scripts/animate124/run-dynamic.sh $GPU run run astronaut-horse "an astronaut, helmet in hand, rides a white horse"

## dynamic fine stage
bash scripts/animate124/run-cn.sh $GPU run run run astronaut-horse "an astronaut, helmet in hand, rides a white horse" "an <token>, helmet in hand, rides a white horse"