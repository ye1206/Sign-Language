from django.db import models


class Image(models.Model):
    title = models.CharField(max_length=200)
    image = models.ImageField(upload_to='images')

    def __str__(self):
        return self.title


class User(models.Model):  # 使用者
    no = models.AutoField(primary_key=True)
    pwd = models.CharField(max_length=20, blank=True, null=True)  # 密碼
    title = models.CharField(max_length=20, blank=True, null=True)
    name = models.CharField(max_length=20, blank=True, null=True)
