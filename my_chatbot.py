"""
       .--.                   .---.
   .---|__|           .-.     |~~~|
.--|===|--|_          |_|     |~~~|--.
|  |===|  |'\     .---!~|  .--|   |--|
|%%|   |  |.'\    |===| |--|%%|   |  |
|%%|   |  |\.'\   |   | |__|  |   |  |
|  |   |  | \  \  |===| |==|  |   |  |
|  |   |__|  \.'\ |   |_|__|  |~~~|__|
|  |===|--|   \.'\|===|~|--|%%|~~~|--|
^--^---'--^    `-'`---^-^--^--^---'--' #Sage_the_librarian

"""

"""
This is a retrieval-based chatbot that recommends books based on user preferences and understands user input using text classification using SitFit library
"""
"""
Most of the code in this file has been adapted from the "24/25 Natural Language Processing for the Creative Industries" unit classes.  
"""





# Import ChatbotBase class
from chatbot_base import ChatbotBase

# Misc
import os
import re
import random as random

# Imports for local data retrieval 
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.downloader.download('vader_lexicon')
from setfit import SetFitModel, SetFitTrainer
import pandas as pd
from sklearn.feature_extraction.text import  TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.preprocessing import normalize


# Surpress PyTorch warning messages bein printed to terminal
import warnings 
warnings.filterwarnings("ignore")



class BookBot(ChatbotBase):
    def __init__ (self, name="Sage"):
        ChatbotBase. __init__(self,name)
        self.name = name
        self.conversation_is_active = True

        self.dataset_path = "datasets/books.csv"
        self.column_names = ["title", "author", "description", "genre",  "pages", "rating", "link" ] 
        self.dataset_separator = ','

        self.df = self.load_books(self.dataset_path, self.dataset_separator, self.column_names)
        self.vectorizer, self.books_matrix = self.books_to_tfidf(self.df)

        # Load in user intent classifier trained model
        self.model = SetFitModel.from_pretrained("ckpt/", local_files_only=True)
        self.sid = SentimentIntensityAnalyzer()

        self.recommended_book = None 

        


        # Respond to Greeting intent
    def greeting(self):
        print( """
      __...--~~~~~-._   _.-~~~~~--...__
    //               `V'               \\ 
   //                 |                 \\ 
  //__...--~~~~~~-._  |  _.-~~~~~~--...__\\ 
 //__.....----~~~~._\ | /_.~~~~----.....__\\
====================\\|//====================
                    `---`
                     """)
        print(f"Hello I am {self.name}, your GoodReads book recommendation bot.")   




     # Respond to Farewell intent
    def farewell(self):
        self.conversation_is_active = False       
        responses = ["happy reading! Goodbye",
                    "Until next time, Enjoy",
                    "Goodbye!", 
                    "Hope this is helpful! Happy reading!"]  
        print(random.choice(responses))



     # Respond to Positive Confirmation intent
    def respond_postive_confirmation(self):
        responses = ['Great!',
                    'Happy to be of service!',
                    'Glad I could help.',
                    'I am glad']           
        print(random.choice(responses))

        # Respond to Negative Confirmation intent
    def respond_negative_confirmation(self):
        response = "I am sorry about that."
        print(response)

        while True:
            user_input = input("Would you like another book recommendation? (yes/no) \n").strip().lower()
            processed_input = self.process_input(user_input)   
            
            if "yes" in user_input: 
                processed_input = input("Can you share with me a book genre you feel like reading, or an author you would like to explore? \n" )  
                return self.respond_book_recommendation(processed_input)  
            
            elif "no" in user_input:  
                return self.farewell()  
            else:
                print("Sorry, I didn't understand that. Please reply with 'yes' or 'no'.")  





     # Respond to Thank you
    def respond_thank_you(self):
        responses = ["You are welcome, happy reading!",
                    "Anytime, happy reading!",
                    "It's my pleasere, enjoy",
                    "Happy to help, enjoy"]   
                
        print(random.choice(responses))




     # Responds to request another book
    def respond_another_book(self, processed_input):
        print("Sure! Let me find another recommendation for you...")

        #this to makes sure the new book is different than the prviously recommended one
        previous_recommendation = self.recommended_book
        filtered_df = self.df[~self.df['title'].str.contains(previous_recommendation, case=False, na=False)]
        filtered_vectorizer, filtered_books_matrix = self.books_to_tfidf(filtered_df)

        rec_author, rec_title, rec_link = self.find_book_recommendation(processed_input, filtered_vectorizer, filtered_books_matrix, filtered_df)
        if rec_author and rec_title:
            self.recommended_book = rec_title
            print(f"Here's a new recommendation: '{rec_title}' by {rec_author}. You can find more details about it here: {rec_link}.")
        else:
            print("I sorry I don't have that. can you give me a different genre or author?")

    
        




    def respond_book_rating(self, rec_title=None):
        rec_title = self.recommended_book
        if rec_title:  
            book = self.df[self.df['title'].str.contains(rec_title, case=False, na=False)]
            if not book.empty:
                rating = book.iloc[0]["rating"]
                print(f"'{book.iloc[0]['title']}' has a rating of {rating}.")
            else:
                print(f"Sorry, I couldn't find the rating for the book titled '{rec_title}'.")
        else:
            print("No book has been recommended yet. Please ask for a recommendation first.")



    

     # Responds to Request book pages
    def respond_book_pages(self, rec_title=None):
            rec_title = self.recommended_book
            book = self.df[self.df['title'].str.contains(rec_title, case=False, na=False)]
            if not book.empty:
                pages = book.iloc[0]["pages"]
                print(f"It has {pages} pages.")
            else:
                print(f"Sorry, I couldn't find the number of page for the book titled '{rec_title}'.")




     # Responds to Request book description
    def respond_book_description(self, rec_title=None):
        rec_title = self.recommended_book
        book = self.df[self.df['title'].str.contains(rec_title, case=False, na=False)]
        if not book.empty:
            description = book.iloc[0]["description"]
            print(f"Here's the description of '{book.iloc[0]['title']}': {description}")
        else:
            print(f"Sorry, I couldn't find a description for the book titled '{rec_title}'.")



     # Responds to Request book link
    def respond_book_link(self, rec_title=None):
        rec_title = self.recommended_book
        book = self.df[self.df['title'].str.contains(rec_title, case=False, na=False)]
        if not book.empty:
            link = book.iloc[0]["link"]
            print(f"Here's the Goodreads link of '{book.iloc[0]['title']}': {link}")
        else:
            print(f"Sorry, I couldn't find a link for the book titled '{rec_title}'.")





    # Responds if the input is not understood
    def did_not_understand(self,processed_input):
        response = "I am sorry, I did not understand that."
        print(response)

        while True:
            user_input2 = input("Would you like a book recommendation? (yes/no) \n").strip().lower()
            processed_input = self.process_input(user_input2)   
            
            if "yes" in user_input2: 
                processed_input = input("Can you share with me a book genre you feel like reading, or an author you would like to explore? \n" )  
                return self.respond_book_recommendation(processed_input)  
            
            elif "no" in user_input2:  
                return self.farewell()  
            else:
                print("Sorry, I didn't understand that. Please reply with 'yes' or 'no'.") 







     # this function loads the books dataset into a pandas data frame and cleans it 
     # This function is generated with the assistance of ChatGPT by OpenAI; then modified for specific use by Iman Ahmed
    
    def load_books(self, file_path, separator, column_names):
        try:
            df = pd.read_csv(file_path, sep=separator, usecols=column_names)

            for col in ["title", "author", "description", "genre"]:
                df[col] = df[col].str.lower().str.replace('[^a-z0-9\s,]', '', regex=True) #you need to keep commas and numbers

            # because the genre cell in the dataset has multible genres spereted with a comma
            df["genre"] = df["genre"].str.split(',')  

            # Preprocess genre column to clean and standardize genre lists
            def clean_genres(genres):
                if isinstance(genres, list):
                     # Strip whitespace, normalize capitalization, and deduplicate
                    genres = [genre.strip().title() for genre in genres]
                    return list(set(genres))  # Remove duplicates
                return []  # Handle missing genres

            df["genre"] = df["genre"].apply(clean_genres)
            return df
        
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return pd.DataFrame()


 


     # this function takes the book data and prepare a TF-IDF matrix to start the search functionality
    def books_to_tfidf(self, df):
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            #this concatenate relevant columns, converting 'genre' list back into a string
            #this line was written with the assistance of ChatGPT by OpenAI
            text_data = (df["author"] + " " + df["description"] + " " + df["genre"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")).fillna("")
            book_matrix = vectorizer.fit_transform(text_data)
            return vectorizer, book_matrix
        except Exception as e:
            print(f"Error creating TF-IDF matrix: {e}")
            return None, None
        



          # this function takes the user input cleans and process it
    def process_input(self, user_input):
        processed_input = re.sub(r'[^\x00-\x7f]',r'', user_input).lower()
        return processed_input




     # this function is for searching books in the dataset and finds the best match using TF-IDF and cosine similarity.
    def find_book_recommendation(self, input_query, vectorizer, book_matrix, df):
        try:
            input_bow = vectorizer.transform([input_query])
            similarities = cosine_similarity(input_bow, book_matrix).flatten()
            nearest_index = similarities.argmax()

            if similarities[nearest_index] == 0:
                return None, None, None

            author = df.iloc[nearest_index]["author"]
            title = df.iloc[nearest_index]["title"]
            link = df.iloc[nearest_index]["link"]
            return author, title, link
        
        except Exception as e:
            print(f"Error finding book recommendation: {e}")
            return None, None, None




     # This function was updated with the assistance of ChatGPT by OpenAI; then modified for specific use by Iman Ahmed
    def respond_book_recommendation(self, processed_input):
        print("Just a moment let me search my bookshelves... ")

        # Parse processed input to identify genre, author, or rating.
        genre = None
        author = None
        rating = None
        
        # Extract genre, author, and rating from the input.
        author_match = re.search(r"(by|author|written by)\s+([A-Z][a-z]+\s[A-Z][a-z]+)", processed_input, re.IGNORECASE)
        if author_match:
            author = author_match.group(2).strip()

        
        genre_match = re.search(r"(genre|about|in the genre of|type of|mood for|read|like to read|want to read|a)\s+([a-z\s]+(?:novel|book|stories)?)", processed_input, re.IGNORECASE)
        if genre_match:
            genre = genre_match.group(2).strip()
            genre = re.sub(r"(novel|book|stories|a)$", "", genre).strip()
            genre = re.sub(r'\s+', ' ', genre)  # keep spaces between words


        rating_match = re.search(r"(rated|rating|with a rating of|above)\s+([0-9.]+)", processed_input, re.IGNORECASE)
        if rating_match:
            rating = float(rating_match.group(2).strip())
        elif "5 stars" in processed_input.lower():
            rating = 5.0
        
        
        # Filter the dataset based on genre, author, and rating.
        filtered_df = self.df

        if genre:
            print(f"Okay let me search for books in the genre: {genre}")
            
            #filtered_df = filtered_df[ filtered_df['genre'].apply(lambda genres: any(genre.lower() in g.lower() for g in genres) if isinstance(genres, list) else False )]
            #Tokenize and clean the input genre to match against dataset genres
            genre_tokens = set(re.sub(r"[^\w\s]", "", genre.lower()).split())

            # Filter dataset for matching genres
            def genre_match(genres):
                if isinstance(genres, list):
                    for g in genres:
                        genre_words = set(g.lower().split())
                        if genre_tokens & genre_words:  # Find overlapping words
                            return True
                return False

            filtered_df = filtered_df[filtered_df['genre'].apply(genre_match)]


        if author:
            print(f"Okay let me search for books by author: {author}")
            filtered_df = filtered_df[filtered_df['author'].str.contains(author, case=False, na=False)]


        if rating:
            print(f"Okay let me search for books with a rating of {rating} or so")
            filtered_df = filtered_df[filtered_df['rating'] >= rating]


        # Perform a TF-IDF similarity search on the filtered dataset.
        input_query = processed_input.strip()

        if not filtered_df.empty:
            filtered_vectorizer, filtered_books_matrix = self.books_to_tfidf(filtered_df)
            #rec_author, rec_title = self.find_book_recommendation(input_query, filtered_vectorizer, filtered_books_matrix, filtered_df)
            rec_author, rec_title, rec_link = self.find_book_recommendation(input_query, filtered_vectorizer, filtered_books_matrix, filtered_df)


            if rec_author and rec_title:
                self.recommended_book = rec_title

                print(f"I recommend '{rec_title}' by {rec_author}.")
                print(f"Find this book on Goodreads here: {rec_link}")
                print(f"Would you like a description of this book? I can provide you with its rating and number of pages if you want.")
            else:
                print("Sorry, I couldn't find this book are you sure about the spelling?")
        else:
            print(f"I apologize, I can't find a book that matches your request.\nCan you try another keyword, auther or genre please!")





    def generate_response(self, processed_input):
        try:
            output_probs = self.model.predict_proba(processed_input)
            output_probs = output_probs.tolist()
            max_conf = max(output_probs)
            
        

            if max_conf < 0.1:
                print("Low confidence detected. Returning fallback response.")
                return self.did_not_understand(processed_input)

            max_class = output_probs.index(max_conf)
            #For debugging 
            #print(f"Predicted Class: {max_class}")

            match max_class:
                case 0:
                    return self.greeting()
                case 1:
                    return self.farewell()
                case 2:
                    return self.respond_postive_confirmation()
                case 3:
                    return self.respond_negative_confirmation()
                case 4:
                    return self.respond_thank_you()
                case 5:
                    return self.respond_book_recommendation(processed_input)
                case 6:
                    return self.respond_another_book(processed_input)
                case 7:
                    return self.respond_book_rating()
                case 8:
                    return self.respond_book_pages()
                case 9:
                    return self.respond_book_description()
                case 10:
                    return self.respond_book_link()
                # If an exact match is not confirmed, this last case will be used if provided
                case _:
                    return self.did_not_understand(processed_input)

        except Exception as e:
            print(f"Error during generate_response: {e}")
            return self.did_not_understand(processed_input)

    


    # this is the main interaction loop function. it calls the main function of the program(generate_response) so it can give the user a respond to their request and loops if they ask for another reques
    def respond(self, out_message = None):
        if isinstance(out_message, str): 
            print(out_message)

        while self.conversation_is_active:
            received_input = self.receive_input()
            processed_input = self.process_input(received_input)
            response = self.generate_response(processed_input)

            if not self.conversation_is_active:
                break
        return response


    



if __name__ == "__main__":
    Book_Bot1 = BookBot()
    Book_Bot1.greeting()

    response = "Can you share with me a book genre you feel like reading, or an author you would like to explore?"

    while Book_Bot1.conversation_is_active:
        response = Book_Bot1.respond(response)

