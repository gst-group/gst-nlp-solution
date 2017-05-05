# This file is just for demo

def get_website_url():
    return "www.guanshantech.com"

def get_comanpay_name():
    return "guanshantech"

def print_insight365_msg():
    manufacture = get_comanpay_name()
    website = get_website_url()

    output = "insight365 is made by {company}, " \
             "please get more detail information by click link:{url}".\
        format(company=get_comanpay_name(),url=get_website_url())

    return output



