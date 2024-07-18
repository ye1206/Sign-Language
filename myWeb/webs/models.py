from django.db import models
from django import forms
from django.conf import settings

# Create your models here.
class Login(forms.Form):  # 登入
    name = forms.CharField(
        required=False,
        label="account",
        widget=forms.TextInput(
            attrs={"placeholder": "name", "required": True}),
    )
    password = forms.CharField(
        required=False, widget=forms.PasswordInput(attrs={"required": True})
    )


class User(models.Model):
    id = models.AutoField(primary_key=True)  # 使用者編號
    username = models.CharField(
        max_length=100, unique=True, null=True)  # 使用者名稱
    password = models.CharField(max_length=100, blank=True, null=True)  # 使用者密碼
