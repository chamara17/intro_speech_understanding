def next_birthday(date, birthdays):
    '''
    Find the next birthday after the given date.

    @param:
    date - a tuple of two integers specifying (month, day)
    birthdays - a dict mapping from date tuples to lists of names, for example,
      birthdays[(1,10)] = list of all people with birthdays on January 10.

    @return:
    birthday - the next day, after given date, on which somebody has a birthday
    list_of_names - list of all people with birthdays on that date
    '''
    month, day = date
    
    # We loop indefinitely until we find a match
    while True:
        # 1. Advance to the next day
        day += 1
        
        # 2. Handle end of month (Simplified logic from lecture: 
        # just assume max 31 days. Invalid dates won't be in the dict anyway.)
        if day >= 32:
            day = 1
            month += 1
            
        # 3. Handle end of year
        if month >= 13:
            month = 1
            
        # 4. Check if this new date exists in the dictionary
        if (month, day) in birthdays:
            # Found it! Return the date and the list of names
            return (month, day), birthdays[(month, day)]
    
