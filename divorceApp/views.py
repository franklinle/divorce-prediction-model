from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_protect
from .divorce import lr_predict_churn
from .forms import MyForm

def prediction_form(request):
    # Use the form in your view logic
    form = MyForm()
    
    # Check if the form has been submitted
    if request.method == 'POST':
        # Bind the form to the POST data
        form = MyForm(request.POST)

        # Validate the form
        if form.is_valid():
            # Process the form data
            input1 = form.cleaned_data['input1']
            input2 = form.cleaned_data['input2']
            input3 = form.cleaned_data['input3']
            input4 = form.cleaned_data['input4']
            input5 = form.cleaned_data['input5']
            input6 = form.cleaned_data['input6']
            input7 = form.cleaned_data['input7']
            input8 = form.cleaned_data['input8']
            input9 = form.cleaned_data['input9']
            input10 = form.cleaned_data['input10']

            prediction_result = lr_predict_churn(input1, input2, input3, input4, input5, input6, input7, input8, input9, input10)

            # return redirect('success')

    # Render the form template
            return render(request, 'divorceApp/prediction_form.html', {'form': form, 'prediction_result': prediction_result})    
            
    # Render the form template
    return render(request, 'divorceApp/prediction_form.html', {'form': form})