GPU=$1

## textual inversion
bash scripts/textual_inversion/textual_inversion_sd.sh $GPU panda-dance _panda_ panda

## static stage
bash scripts/animate124/run-static.sh $GPU run panda-dance "a high-resolution DSLR image of a panda"

## dynamic corase stage
bash scripts/animate124/run-dynamic.sh $GPU run run panda-dance "a panda is dancing"

## dynamic fine stage
bash scripts/animate124/run-cn.sh $GPU run run run panda-dance "a panda is dancing" "a <token> is dancing"