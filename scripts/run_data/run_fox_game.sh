GPU=$1

## textual inversion
bash scripts/textual_inversion/textual_inversion_sd.sh $GPU fox-game _fox_ fox

## static stage
bash scripts/animate124/run-static.sh $GPU run fox-game "a high-resolution DSLR image of a fox maneuvering a game controller with its paws"

## dynamic corase stage
bash scripts/animate124/run-dynamic.sh $GPU run run fox-game "a fox maneuvering a game controller with its paws"

## dynamic fine stage
bash scripts/animate124/run-cn.sh $GPU run run run fox-game "a fox maneuvering a game controller with its paws" "a <token> maneuvering a game controller with its paws"