import sys

import nltk
import pandas as pd
import regex as re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from nlp.exception import SpamException

nltk.download('stopwords')


class SpamClassifier: 

    def __init__(self,dataset_path): 
        try : 
            self.path = dataset_path
        except Exception as e : 
            print(f"Exception occured due to {e}") 
            raise SpamException(e,sys) from e  
    def get_corpus(self): 
        try : 
            dataset = self.get_dataset()
            stemmar = PorterStemmer()
            stopword_list = stopwords.words('english')
            corpus = []
            for  i in range(0,len(dataset)): 
                sentence = re.sub('[^a-zA-Z0-9]',' ',dataset['message'][i])
                sentence = sentence.lower()
                word_list = sentence.split()
                word = [stemmar.stem(word) for word in word_list if word not in stopword_list]
                corpus.append(' '.join(word))
            return corpus
        except Exception as e : 
            print("Exception occured due to {e}")
            raise SpamException(e,sys) from e 

    def get_dataset(self): 
        try : 
            dataset = pd.read_csv(self.path,sep='\t',header=None)
            dataset.columns=['label','message'] 
            return dataset 
        except Exception as e : 
            print(f"Exception occured due to {e}")
            raise SpamException(e,sys) from e 

class SpamTransformer: 

    def __init__(self,corpus,dataset) -> None:
        try : 
            self.corpus = corpus 
            self.dataset = dataset 
            
        except Exception as e :
            print("Exception occured due to {e}")
            raise SpamException(e,sys) from e 

    def feature_transform(self): 
        try : 
            
            x = self.corpus
            y = pd.get_dummies(self.dataset['label'],drop_first=True)
            x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.20 , random_state=0)
            cv = CountVectorizer(ngram_range=(1,2), max_features=2500)
            x_train = cv.fit_transform(x_train)
            x_test = cv.transform(x_test)
            self.cv = cv 
            rf_bow = RandomForestClassifier()
            rf_bow.fit(x_train,y_train)
            self.rf = rf_bow
        except Exception as e : 
            print("Exception occured due to {e}")
            raise SpamException(e,sys) from e 

        
    def predict(self,x): 
        try : 
            x = self.cv.transform(x)
            y_pred = self.rf.predict(x)
            return y_pred
        except Exception as e : 
            raise SpamException(e,sys) from e 
 

    
    

        
        
