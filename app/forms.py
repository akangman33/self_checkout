from django.forms import ModelForm, TextInput
from datetime import date
from .models import Record

class RecordForm(ModelForm):
    class Meta:
        model = Record
        #fields = ['date', 'description', 'category', 'cash', 'balance_type']
        fields = ['date', 'category', 'cash', 'cnt']
        #widgets允許某個欄位進行客製化處理
        widgets = {
            'date':TextInput(
                attrs={
                    'id':'datepicker1',
                    'value':date.today().strftime('%Y-%m-%d')
                }
            )
        }