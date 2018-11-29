img_size = 224 # resize images to 224*224 px

emotions = ['Neutral', 'Positive','Negative','Anxiety']
nb_emotions = len(emotions)

def get_emotion_class(valence, arousal):
    emotion = None
    if valence==0:
        # Neutral emotion 
        emotion = 0
    elif valence==1:
        # Positive emotion
        emotion = 1
    elif valence==-1:
        if arousal >=0:
            # Negative emotion
            emotion = 2
        else:
            # Anxiety
            emotion = 3
            
    return emotion