# Import to read dataset
import pandas as pd
# Import Vectorizer Function
from sklearn.feature_extraction.text import TfidfVectorizer

# Import Similarity Function
from sklearn.metrics.pairwise import cosine_similarity

# Import For Getting Similar Named Input
from difflib import get_close_matches

class SongRecommender():
    def __init__(self,similar_songs_dict):
        self.similar_songs_dict = similar_songs_dict

    # Print output for the user
    def print_similar_songs(self,input_song,recommended_songs):
        print("\n")
        print (f"Recommended songs for '{input_song}' are: \n")

        for i in range(len(recommended_songs)):
            print(f"{i+1}: ", end=' ')
            print(f" '{recommended_songs[i][1]}' by '{recommended_songs[i][2]}' with similarity score of {recommended_songs[i][0]}")
            print("---")
    
    # Get recommended songs from the dataset
    def get_recommendation(self,recommendation_dict):
        self.song = recommendation_dict['song']
        self.number_of_songs = recommendation_dict['no_of_songs']

        self.recommended_songs = self.similar_songs_dict[self.song][-self.number_of_songs:]

        self.recommended_songs.reverse()
        
        self.print_similar_songs(input_song=self.song,recommended_songs=self.recommended_songs)


# Function to clean dataset
def clean_data_set():
    # Loading final data without 'link' column
    songs = pd.read_csv('dataset/songs_0_20.csv')
    songs = songs.sample(n=20000).drop('link',axis=1).reset_index(drop=True)

    # Cleaning data, removing \n and errors during scraping lyrics
    songs['text'] = songs['text'].str.replace(r'\n','')
    songs = songs[~songs['text'].str.contains('>503 Service Temporarily Unavailable')]
    return songs 

# Function that stores similar songs for every song
def find_similar_songs():
    
    songs = clean_data_set()
    # Calculating TF-IDF Score and stroing in matrix form
        # Creating vectorizer object
    tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
        # Storing scores in a matrix
    vector_matrix = tfidf.fit_transform(songs['text'])

    # Calculating cosine similarity for every single lyrics from the matrix
    cosine_similarity_scores = cosine_similarity(vector_matrix)

    similar_songs = {}
    # Creating dictionary to store 20 most similar songs for each song
    for i in range(len(cosine_similarity_scores)):
        # Sorting indices that represent similar song for index i
        similar_indices = cosine_similarity_scores[i].argsort()[-20:-1]
        # Storing similar songs for the given song i, excluding the song itself
        similar_songs[str(songs['song'].iloc[i])] = [(cosine_similarity_scores[i][x], songs['song'].iloc[x],songs['artist'].iloc[x])for x in similar_indices][1:]
    return similar_songs

# Getting dictionary with similar songs
similar_songs_dict = find_similar_songs()
def recommend(song, number_of_songs):
    sr = SongRecommender(similar_songs_dict)
    recommendation_dict = {
        "song": song,
        "no_of_songs": number_of_songs
    }
    sr.get_recommendation(recommendation_dict=recommendation_dict)

# Main function to take input and pass it to recommender function
if __name__ == '__main__':
    dataset = clean_data_set()

    loop = True
    while loop:
        input_song = input("Enter a song: ")
        if input_song.strip() != None:
            input_similar_songs = get_close_matches(input_song, similar_songs_dict.keys())
            
            if not input_similar_songs:
                print("Songs similar to this name does not exist in the dataset. Please try entering a different song.")
            else:
                input_similar_song = input_similar_songs[0]
                song_index_list = dataset.index[dataset['song'] == input_similar_song].tolist()
                song = dataset['song'].loc[song_index_list[0]]
                
                recommend(song=song,number_of_songs=5)
                loop = False
        else:
            print("Please enter a name of a song.")
            print("-------------------------------")
            continue