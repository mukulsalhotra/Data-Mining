# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:30:31 2017

@author: ShivamMaurya

modified the json_to_csv_converter.py to work with python 3
"""
import json
import csv
import collections

def get_column_names(line_contents, parent_key=''):
    column_names = []
    for k, v in line_contents.items():
        column_name = "{0}.{1}".format(parent_key, k) if parent_key else k
        if isinstance(v, collections.MutableMapping):
            column_names.extend(
                    get_column_names(v, column_name).items()
                    )
        else:
            column_names.append((column_name, v))
    return dict(column_names)

def get_superset_of_column_names_from_file(json_file_path):
    column_names = set()
    with open(json_file_path, encoding="utf8") as fin:
        for line in fin:
            line_contents = json.loads(line)
            column_names.update(
                    set(get_column_names(line_contents).keys())
                    )
    return column_names

def get_nested_value(d, key):
    #Return a dictionary item given a dictionary `d` and a flattened key from `get_column_names`.
    if '.' not in key:
        if key not in d:
            return None
        return d[key]
    base_key, sub_key = key.split('.', 1)
    if base_key not in d:
        return None
    sub_dict = d[base_key]
    return get_nested_value(sub_dict, sub_key)

def get_row(line_contents, column_names):
    row = []
    for column_name in column_names:
        line_value = get_nested_value(
                        line_contents,
                        column_name,
                        )
        if isinstance(line_value, str):
            row.append('{0}'.format(line_value.encode('utf-8')))
        elif line_value is not None:
            row.append('{0}'.format(line_value))
        else:
            row.append('')
    return row

def read_and_write_file(json_file_path, csv_file_path, column_names):
    """Read in the json dataset file and write it out to a csv file, given the column names."""
    with open(csv_file_path, 'w') as fout:
        csv_file = csv.writer(fout)
        csv_file.writerow(list(column_names))
        with open(json_file_path,encoding="utf8") as fin:
            for line in fin:
                line_contents = json.loads(line)
                csv_file.writerow(get_row(line_contents, column_names))
                

if __name__ == '__main__':
    ext_csv = '.csv'
    ext_json = '.json'

    folder_json = '../data/json/'
    folder_csv = '../data/csv/'

    file_tip = 'yelp_academic_dataset_tip'
    file_user = 'yelp_academic_dataset_user'
    file_business = 'yelp_academic_dataset_business'
    file_review = 'yelp_academic_dataset_review'
    file_checkin = 'yelp_academic_dataset_checkin'
    
    #Business Data Set
    json_file = '{0}{1}{2}'.format(folder_json, file_business, ext_json)
    csv_file = '{0}{1}{2}'.format(folder_csv, file_business, ext_csv)
    print('{0} -> {1}'.format(json_file, csv_file))
    
    column_names = get_superset_of_column_names_from_file(json_file)
#    print(column_names)
    read_and_write_file(json_file, csv_file, column_names)
#    print(csv_file.title())

    '''
    Convert the review file
    '''

    json_file = '{0}{1}{2}'.format(folder_json, file_review, ext_json)
    csv_file = '{0}{1}{2}'.format(folder_csv, file_review, ext_csv)

    print( '{0} -> {1}'.format(json_file, csv_file))
    column_names = get_superset_of_column_names_from_file(json_file)
    read_and_write_file(json_file, csv_file, column_names)
    
    '''
    Convert the user file
    '''

    json_file = '{0}{1}{2}'.format(folder_json, file_user, ext_json)
    csv_file = '{0}{1}{2}'.format(folder_csv, file_user, ext_csv)

    print( '{0} -> {1}'.format(json_file, csv_file))
    column_names = get_superset_of_column_names_from_file(json_file)
    read_and_write_file(json_file, csv_file, column_names)
    
    '''
    Convert the tip file
    '''

    json_file = '{0}{1}{2}'.format(folder_json, file_tip, ext_json)
    csv_file = '{0}{1}{2}'.format(folder_csv, file_tip, ext_csv)

    print( '{0} -> {1}'.format(json_file, csv_file))
    column_names = get_superset_of_column_names_from_file(json_file)
    read_and_write_file(json_file, csv_file, column_names)
    
    '''
    Convert the checkin file
    '''

    json_file = '{0}{1}{2}'.format(folder_json, file_checkin, ext_json)
    csv_file = '{0}{1}{2}'.format(folder_csv, file_checkin, ext_csv)

    print( '{0} -> {1}'.format(json_file, csv_file))
    column_names = get_superset_of_column_names_from_file(json_file)
    read_and_write_file(json_file, csv_file, column_names)

