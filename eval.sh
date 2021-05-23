wget https://www.dropbox.com/s/ir7sweersamq834/processed_data.pkl
mv processed_data.pkl ./data
python3 disentangle.py esim4 --model ./data/esim_joint.pt.model --test ./data/test/*annotation.txt --hidden 128  --test-start 1000 --test-end 2000  --word-vectors glove-ubuntu.txt > cluster.out
python3 task-4-evaluation.py --gold ./data/test/*anno* --auto cluster.out