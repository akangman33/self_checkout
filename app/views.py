from django.shortcuts import render, HttpResponse, redirect
from .models import Record, Category
from .forms import RecordForm
from django.contrib.auth.decorators import login_required


# Create your views here.
@login_required
def hello(request):
    return render(request,'app/hello.html',{}) #記得去setting註冊app雖然已經startapp!!
@login_required
def frontpage(request):
    # Model Form operation
    user = request.user #
    record_form = RecordForm()
    records = Record.objects.filter(user=user)#
    cash_list = [record.cash*record.cnt for record in records]
    type_cnt = 0
    type_list = []
    object_cnt =0
    for record in records:
        object_cnt+= record.cnt
        if not (record.category in type_list):
            type_list.append(record.category)
            type_cnt +=1

    total_cash = sum(cash_list) if len(cash_list)!=0 else 0

    return render(request, 'app/index.html', locals())
@login_required
def settings(request):
    user = request.user #
    categories = Category.objects.filter(user=user)#
    return render(request, 'app/settings.html', locals())

# Template form operation
@login_required
def addCategory(request):
    if request.method == 'POST':
        user = request.user #
        posted_data = request.POST #request 的屬性.POST 把POST的資料傳進來(dictionary的格式)
        category = posted_data["add_category"]
        Category.objects.get_or_create(category=category, user=user)#
    return redirect('/settings')
@login_required
def deleteCategory(request, category):
    user = request.user#
    Category.objects.filter(category=category, user=user).delete()#
    return redirect('/settings')
@login_required
def addRecord(request):
    if request.method == 'POST':
        user=request.user#
        #ModelForm的威力!! 賦予RecordForm一個dictionary(request.POST)，並存成一個form的物件
        #把form的值填滿，並對form進行檢測(檢測需另外定義)，本例沒有使用
        form = RecordForm(request.POST)
        if form.is_valid():
            #form.save() #original
            record = form.save(commit=False)
            record.user = user
            record.save()
    return redirect('/')
@login_required
def deleteRecord(request):
    if request.method == 'POST':
        id = request.POST['delete_val']
        Record.objects.filter(id=id).delete()
    return redirect('/')

