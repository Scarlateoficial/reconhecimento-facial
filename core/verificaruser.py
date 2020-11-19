from requests import get, Request, post
import json

def auth_user(matricula, email, password):
    busca = get("http://localhost:8000/api/usuarios")
    dados = busca.json()
    for dado in dados:
        if matricula == dado['matricula']:
            if email == dado['email']:
                if password == dado['password']:
                    return dado['id']
    
    return False

def coleta_dados(id):
    busca = get("http://localhost:8000/api/usuarios")
    dados = busca.json()
    for dado in dados:
        if id == dado['id']:
            return { 
                "nome": dado['nome'],
                "matricula": dado['matricula'],
                "email": dado['email'],
                "cargo": dado['cargo'],
            }
    
    return False