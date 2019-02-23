import os
import sys
import django
import time
import datetime

pathname = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, pathname)
sys.path.insert(0, os.path.abspath(os.path.join(pathname, '..')))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mysite.settings")
django.setup()

from app.models import Category, Record
from django.contrib.auth.models import User

if __name__ == "__main__":
    user = User.objects.create_user('john', 'lennon@thebeatles.com', 'johnpassword')
    a = User.objects.get(username='john')
    b = a.id
    print(b)
    words = [['豆漿燕麥', '1', '40'], ['奶綠', '1', '25'], ['奶綠', '1', '25'], ['奶綠', '1', '25'], ['奶綠', '1', '25']]
    for word in words:
        name = word[0]
        count = word[1]
        price = word[2]
        c = Category(category=name, user_id=b)
        unix = time.time()
        date = str(datetime.datetime.fromtimestamp(unix).strftime('%Y-%m-%d'))
        d = Record(date=date, category=name, cash=price, cnt=count, user_id=b)
        c.save()
        d.save()

    pass