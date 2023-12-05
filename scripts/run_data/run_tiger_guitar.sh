GPU=$1

## textual inversion
bash scripts/textual_inversion/textual_inversion_sd.sh $GPU tiger-guitar _tiger_ tiger

## static stage
bash scripts/animate124/run-static.sh $GPU run tiger-guitar "a high-resolution DSLR image of a full-bodied tiger standing on its hind legs, confidently playing an acoustic guitar"

## dynamic corase stage
bash scripts/animate124/run-dynamic.sh $GPU run run tiger-guitar "a full-bodied tiger standing on its hind legs, confidently playing an acoustic guitar"

## dynamic fine stage
bash scripts/animate124/run-cn.sh $GPU run run run tiger-guitar "a full-bodied tiger standing on its hind legs, confidently playing an acoustic guitar" "a <token> standing on its hind legs, confidently playing an acoustic guitar"