# openGL
Um projeto para entender o funcionamento basico do OpenGL.

## Ambiente recomendado
Use Python 3.12.

O projeto agora usa `arcade` para a janela, input e loop principal, mantendo o rendering em OpenGL via `PyOpenGL`.

No Python 3.14, `numpy==1.24.4` nao possui wheel para `cp314` no Windows e `pymunk` pode cair em compilacao local. Isso costuma exigir o Microsoft C++ Build Tools.

## Instalacao
1. Crie a virtualenv com `py -3.12 -m venv .venv`.
2. Ative a virtualenv com `.\.venv\Scripts\Activate.ps1`.
3. Atualize as ferramentas base com `python -m pip install -U pip setuptools wheel`.
4. Instale as dependencias com `python -m pip install -r requirements.txt`.

## Exemplo
![alt text](https://github.com/BrunoVieiraBarbosa/openGL/blob/main/img/light.png)
