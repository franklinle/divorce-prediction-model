from django import forms

class MyForm(forms.Form):
    INPUT_CHOICES = [
        (0, 'Always'),
        (1, 'Frequently'),
        (2, 'Averagely'),
        (3, 'Sometimes'),
        (4, 'Never'),
    ]
    input1 = forms.ChoiceField(label='Input 1', choices=INPUT_CHOICES, widget=forms.RadioSelect)
    input2 = forms.ChoiceField(label='Input 2', choices=INPUT_CHOICES, widget=forms.RadioSelect)
    input3 = forms.ChoiceField(label='Input 3', choices=INPUT_CHOICES, widget=forms.RadioSelect)
    input4 = forms.ChoiceField(label='Input 4', choices=INPUT_CHOICES, widget=forms.RadioSelect)
    input5 = forms.ChoiceField(label='Input 5', choices=INPUT_CHOICES, widget=forms.RadioSelect)
    input6 = forms.ChoiceField(label='Input 6', choices=INPUT_CHOICES, widget=forms.RadioSelect)
    input7 = forms.ChoiceField(label='Input 7', choices=INPUT_CHOICES, widget=forms.RadioSelect)
    input8 = forms.ChoiceField(label='Input 8', choices=INPUT_CHOICES, widget=forms.RadioSelect)
    input9 = forms.ChoiceField(label='Input 9', choices=INPUT_CHOICES, widget=forms.RadioSelect)
    input10 = forms.ChoiceField(label='Input 10', choices=INPUT_CHOICES, widget=forms.RadioSelect)
