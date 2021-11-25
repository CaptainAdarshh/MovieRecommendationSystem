from django import forms

class MovieRecommendForm(forms.Form):
    input = forms.Textarea()