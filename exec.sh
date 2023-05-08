# vitstr
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_19 --img_w 224 --img_h 224 --action predict --encoder ViTSTR

# TRBA
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_13 --action predict --trans TPS --encoder ResNet --SequenceModeling BiLSTM --decoder SeqAttn

#STAR-Net
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_12 --action predict --trans TPS --encoder ResNet --SequenceModeling BiLSTM --decoder CTC


#Rosetta
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_11 --action predict --encoder ResNet --decoder CTC

# GRCNN
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_10 --action predict --encoder GRCNN --SequenceModeling BiLSTM --decoder CTC

# RARE
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_9 --action predict --trans TPS --encoder VGG --SequenceModeling BiLSTM --decoder SeqAttn

# R2AM
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_8 --action predict  --encoder GRCNN --decoder SeqAttn

# CRNN
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_6 --action predict  --encoder VGG --SequenceModeling BiLSTM --decoder CTC

# SVTR_tiny
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_5  --action predict  --encoder SVTR_T --decoder CTC

# SVTR_tiny + LM(3)
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_17 --action predict  --encoder SVTR_T --decoder LM --language_module BCN

# TPS + SVTR_tiny + LM(3)
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_18  --action predict --trans TPS --encoder SVTR_T --decoder LM --language_module BCN
