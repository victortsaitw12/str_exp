# vitstr
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_19 --img_w 224 --img_h 224 --action predict --trans None --encoder ViTSTR --decoder None --language_module None

# TRBA
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_13 --action predict --trans TPS --encoder ResNet --SequenceModeling BiLSTM --decoder SeqAttn --language_module None

#STAR-Net
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_12 --action predict --trans TPS --encoder ResNet --SequenceModeling BiLSTM --decoder CTC --language_module None


#Rosetta
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_11 --action predict --trans None --encoder ResNet --decoder CTC --language_module None

# GRCNN
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_10 --action predict --trans None --encoder GRCNN --SequenceModeling BiLSTM --decoder CTC --language_module None

# RARE
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_9 --action predict --trans TPS --encoder VGG --SequenceModeling BiLSTM --decoder SeqAttn --language_module None

# R2AM
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_8 --action predict  --trans None --encoder GRCNN --decoder SeqAttn --language_module None

# CRNN
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_6 --action predict   --trans None --encoder VGG --SequenceModeling BiLSTM --decoder CTC --language_module None

# SVTR_tiny
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_5  --action predict  --trans None --encoder SVTR_T --decoder CTC --language_module None

# SVTR_tiny + LM(3)
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_17 --action predict  --trans None --encoder SVTR_T --decoder LM --language_module BCN

# TPS + SVTR_tiny + LM(3)
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_18  --action predict --trans TPS --encoder SVTR_T --decoder LM --language_module BCN

# TPS + SVTR_Large + LM(3)
& C:/Users/victor/anaconda3/python.exe c:/Users/victor/Desktop/experiment/src/main.py --save_path C:\Users\victor\Desktop\exp_id_22  --action predict --trans TPS --encoder SVTR_T --decoder LM --language_module BCN
