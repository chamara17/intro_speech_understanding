def words2characters(words):
    """
    This function converts a list of words into a list of characters.

    @param:
    words - a list of words

    @return:
    characters - a list of characters

    Every element of "words" should be converted to a str, then split into
    characters, each of which is separately appended to "characters." For 
    example, if words==['hello', 1.234, True], then characters should be
    ['h', 'e', 'l', 'l', 'o', '1', '.', '2', '3', '4', 'T', 'r', 'u', 'e']
    """
    characters = []             # 1. Create an empty list
    for item in words:          # 2. Loop through every item in the input list
        string_item = str(item) # 3. Convert the item to a string
        for char in string_item: # 4. Loop through every character in that string
            characters.append(char) # 5. Add the character to the new list
            
    return characters           # 6. Return the result