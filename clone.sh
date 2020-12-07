rm -rf ./data_collection/gym_duckietown || true
rm -rf ./requirements.txt || true
(git clone https://github.com/Velythyl/gym-duckietown.git -b object_detection temp  || git clone git@github.com:Velythyl/gym-duckietown.git -b object_detection temp) && cp temp/requirements.txt ./requirements.txt && cp -r ./temp/src/gym_duckietown ./data_collection/gym_duckietown && rm -rf temp
