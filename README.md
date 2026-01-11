# Gachamii Dialogue Generator

- Dialogue generator changed for use in Gachamii mobile game
- Forked from [equalo-official](https://github.com/equalo-official/animalese-generator) and using original sounds from that repo
- Rebuilt with python 3.13.3

## Requirements
```sh
pip install -r requirements.txt
```

## Animal-crossing-esque dialogue creator
```sh
python animalese.py "Enter dialogue to convert here" --pitch {low,medium,default,high) --tempo {0.0-~2.0} --outfile {ex. my_dialogue.wav}
# pitch changes voice pitch; default is "default"
# tempo > 1.0 => faster speech (shorter); default is 0.75
```

## Kirby-fy an existing mp3 or wav file
```sh
python kirbify.py {input file path/name; ex. input_sound.mp3} --octaves {0.0 to ~2.0} --tempo {0.0-~2.0} --out {outfile path/name.wav; ex. kirbified_sound.wav}
# tempo > 1.0 => faster speech (shorter); default is 1.0
# octaves > 1.0 => higher-pitched; default is 1.0
```


