# import spacy with the appropriate language model
import spacy

nlp = spacy.load('en_core_web_md')

# read the movies.txt file and convert to a list
f = open("movies.txt", "r")
movies = f.readlines()

# create variable planet hulk to store its description
planet_hulk = "Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, " \
              "the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk " \
              "can live in peace. Unfortunately, Hulk land on the planet Sakaar where he is sold into slavery " \
              "and trained as a gladiator."

# create two empty lists
# one will store the movie titles and the other will store the movie descriptions
movie_titles = []
movie_descriptions = []

# for each line of the list version of the movies text files
#   strip the line of the new line character and split by the colon
#   store the movie title at index 0
#   store the movie description at index 1
#   append these to their appropriate empty list from above
for line in movies:
    description = line.strip("\n").split(":")
    comp_movie = description[0]
    comp_descr = description[1]
    movie_titles.append(comp_movie)
    movie_descriptions.append(comp_descr)

# close the text file
f.close()

# say the model that we are comparing the movie description to is that of planet hulk
model_movie = nlp(planet_hulk)


# define a function to return which movie a user would watch next if they have watched
# based on having watched Planet Hulk
# taking the description as the parameter
def movie_similarity(descr):
    # create varialbles to store the most similar movie description to planet hulk, the index of that movie, and
    # a counter variable and initialise all to 0
    most_similar = 0.00
    desc_index = 0
    counter = 0

    # for each description in the movie descriptions list
    #   compare them to the planet hulk description model
    #   if the similarity between that movie and captain hulk is more than the most similar stored
    #   the most similar now becomes that movie
    #   the index of that movie description is equal to the counter
    # increment the counter by 1
    for desc in movie_descriptions:
        similarity = nlp(desc).similarity(model_movie)
        if similarity > most_similar:
            most_similar = similarity
            desc_index = counter
        counter += 1

    # the suggested movie is located in the movie titles list at the same index position as the most similar
    # movie description
    suggested_movie = movie_titles[desc_index]

    # return the suggested movie
    return suggested_movie


# create a variable call the above function
ans = movie_similarity(movie_descriptions)

# print the relevant message with the movie that the user should watch next
print(f"Based on your preferences you should watch {ans} next")
