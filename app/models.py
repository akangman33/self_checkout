from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Category(models.Model):
    category = models.CharField(max_length=20)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    
    def __str__(self):
        return self.category

class Record(models.Model):
    date = models.DateField()
    category = models.CharField(max_length=20)
    cash = models.IntegerField()
    cnt = models.IntegerField(default=1)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    def __str__(self):
        return str(self.category)