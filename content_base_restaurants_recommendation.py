import math
import pandas as pd
import toolz
import numpy as np
from sklearn import linear_model

categories_path = "data/restaurants/restaurants_categories.csv"
items_path = "data/restaurants/items.txt"
rating_train_path = "data/restaurants/restaurants_training.txt"
rating_test_path = "data/restaurants/restaurants_test.txt"

def create_item_categories():
    items_data = pd.read_csv(items_path, delimiter="::", header=None, engine='python')
    items_data.columns = ['itemid', 'categories']

    # data cleaning (remove the records that are greater than 100 chars or have HREF )
    items_data = items_data[
        (items_data['categories'].str.len() < 100) & (~items_data['categories'].str.contains("a href"))]

    items_dictionary = dict(zip(
        items_data['itemid'].values.tolist(),
        [v.replace("Fast Food", "FastFood_").
             replace(" Food", " ").  # remove word 'Food'
             replace("Food ", " ").
             replace("  ", " ").  # remove extra blanks
             replace("  ", " ").
             strip()  # trim
         for v in items_data['categories'].values.tolist()]))

    return items_dictionary


def get_categories_list(x):
    list_split = x.split(" ")
    if list_split:
        return categories_to_ids(x, [list_split[0]], list_split[1:], set())
    return list()


def categories_to_ids(orig_str, curr_list, rest_list, id_set):
    #decode categories to IDs

    categories_data = pd.read_csv(categories_path, header=None, sep='@')

    categories_dictionary = dict(
        zip(categories_data[0].values.tolist(), range(len(categories_data[0].values.tolist()))))

    key = " ".join(curr_list)

    if (key in categories_dictionary and not rest_list):
        return id_set | {categories_dictionary[key]}  # this is the last category

    if (key not in categories_dictionary and not rest_list):
        print("key not found: " + key + "; orig_str:" + orig_str)
        return id_set

    # check if the category can be expanded by next token (greedy algorithm)
    next_key = " ".join(curr_list + rest_list[0:1])
    if (key in categories_dictionary and next_key not in categories_dictionary and rest_list[0] not in ['&', 'and']):
        # there is also match for next token
        return categories_to_ids(orig_str, rest_list[0:1], rest_list[1:], id_set | {categories_dictionary[key]})
    else:
        # no match for next token - using current category
        return categories_to_ids(orig_str, curr_list + rest_list[0:1], rest_list[1:], id_set)


def create_category_frame(item_category_dictionary):
    items_list = list()
    category_list = list()
    for k, value in item_category_dictionary.items():
        for v in value:
            items_list.append(k)
            category_list.append(v)
    item_category_df = pd.DataFrame({'itemid': items_list, 'categoryid': category_list})
    # add value column
    import numpy as np
    item_category_df['value'] = pd.Series(np.ones((len(items_list),)), index=item_category_df.index)

    #create frame with itemids as rows and categories as columns
    return item_category_df.pivot(index='itemid', columns='categoryid', values='value').fillna(0)


def normalize_category_frame(category_frame, category_idf):
    # normalize features
    number_of_features = category_frame.sum(axis=1)
    category_frame = category_frame.apply(lambda r: r.apply(
        lambda v: v / np.sqrt(number_of_features[r.name]) if (v > 0) else v), axis=1)

    return category_frame.apply(lambda r: np.multiply(r, category_idf), axis=1)


def create_category_idf(category_frame):
    return category_frame.sum(axis=0).map(lambda x: math.log10(category_frame.shape[0] / x))


def train_linear_model(rating_train_data_with_categories):
    user_profile_model_dict = dict()
    user_profile_avg_dict = dict()

    for name, group in rating_train_data_with_categories.groupby('userid'):
        rating = group[['rating']]
        df = group.drop('userid', axis=1).drop('rating', axis=1).drop('itemid', axis=1)

        regr = linear_model.Lasso(alpha=1)
        regr.fit(df, rating)

        user_profile_model_dict[name] = np.append(regr.coef_, regr.intercept_)
        user_profile_avg_dict[name] = np.average(rating)

    return (user_profile_model_dict, user_profile_avg_dict)


def calculate_rmse(rating_test_data, user_profile_model_dict, user_profile_avg_dict, categories_frame):
    rmse = list()
    for index, row in rating_test_data.iterrows():

        if row['userid'] in user_profile_model_dict:

            if row['itemid'] in categories_frame.index:
                res = np.dot(np.append(categories_frame.loc[row['itemid']].as_matrix(), 1),
                             user_profile_model_dict[row['userid']])
            else:
                #             print (row['itemid'])
                res = user_profile_avg_dict[row['userid']]

            rmse.append(math.pow(row['rating'] - res, 2))

    return math.sqrt(sum(rmse) / float(len(rmse)))


def main():
    # create item -> category map
    categories = create_item_categories()

    #decode categories
    item_category_dictionary = toolz.valmap(get_categories_list, categories)

    #create data frame: rows items, cols categories
    categories_frame = create_category_frame(item_category_dictionary)

    #create IDF vestor for categories
    category_idf = create_category_idf(categories_frame)

    #normalize values by square root of categories count and by IDF
    category_frame_normalized = normalize_category_frame(categories_frame, category_idf)

    #read train and test rating data
    rating_train_data = pd.read_csv(rating_train_path, delimiter="::", header=None, engine='python')
    rating_train_data.columns = ['userid', 'itemid', 'rating']
    rating_test_data = pd.read_csv(rating_test_path, delimiter="::", header=None, engine='python')
    rating_test_data.columns = ['userid', 'itemid', 'rating']

    #join rating data with categories data
    rating_train_data_with_categories = pd.merge(left=rating_train_data, right=category_frame_normalized,
                                                 left_on='itemid', right_index=True)

    #build linear model
    (user_profile_model_dict, user_profile_avg_dict) = train_linear_model(rating_train_data_with_categories)

    #calculate RSME
    rmse = calculate_rmse(rating_test_data, user_profile_model_dict, user_profile_avg_dict, categories_frame)

    print("RSME for restaurant recommendation is ",rmse)


if __name__ == "__main__":
    main()
