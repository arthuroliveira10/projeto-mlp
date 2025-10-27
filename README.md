# MLP para XOR - Perceptron Multicamadas

Este projeto implementa uma **Multi-Layer Perceptron (MLP)** para resolver o problema do XOR usando backpropagation.

## 📋 Descrição

O problema XOR (OU Exclusivo) é um problema clássico que demonstra a importância de camadas ocultas em redes neurais. Uma rede neural de camada única não consegue resolver o XOR, mas com uma camada oculta é possível.

## 🔧 Funcionamento do Algoritmo

### Arquitetura da Rede
- **Camada de entrada**: 2 neurônios (x1 e x2)
- **Camada oculta**: 2 neurônios (h1 e h2) com função de ativação sigmoid
- **Camada de saída**: 1 neurônio (y) com função de ativação sigmoid

### Estrutura dos Pesos
```
Entradas (x1, x2) → Camada Oculta (h1, h2) → Saída (y)

Pesos da camada oculta:
- w11: x1 → h1
- w12: x1 → h2  
- w21: x2 → h1
- w22: x2 → h2

Pesos da camada de saída:
- wh1: h1 → y
- wh2: h2 → y

Biases:
- b1: bias para h1
- b2: bias para h2
- b3: bias para y
```

### Processo de Treinamento

1. **Forward Propagation**: Calcula as saídas da rede
   ```
   h1 = sigmoid(x1 * w11 + x2 * w21 + b1)
   h2 = sigmoid(x1 * w12 + x2 * w22 + b2)
   y = sigmoid(h1 * wh1 + h2 * wh2 + b3)
   ```

2. **Backward Propagation**: Calcula os gradientes e atualiza os pesos
   - Calcula o erro: `erro = saída_desejada - saída_calculada`
   - Propaga o erro de trás para frente usando a regra da cadeia
   - Atualiza os pesos usando a fórmula: `peso_novo = peso_antigo + taxa_aprendizado * gradiente`

3. **Repetição**: O processo é repetido por 50.000 épocas

### Função Sigmoid
```python
sigmoid(x) = 1 / (1 + e^(-x))
```

A função sigmoid é usada para:
- Garantir que as saídas fiquem entre 0 e 1
- Permitir a derivação necessária para o backpropagation

## 📊 Tabela XOR

| Entrada | Saída Esperada |
|---------|----------------|
| [0, 0]  | 0              |
| [0, 1]  | 1              |
| [1, 0]  | 1              |
| [1, 1]  | 0              |

## 🚀 Como Executar

### Pré-requisitos
- Python 3.x
- NumPy

### Instalação
```bash
pip install numpy
```

### Execução
```bash
python mlp_xor.py
```

### Resultado Esperado
```
Resultados:
[0, 0] -> 0
[0, 1] -> 1
[1, 0] -> 1
[1, 1] -> 0
```

## ⚙️ Parâmetros

- **Taxa de Aprendizado (alpha)**: 0.05
- **Número de Épocas**: 50.000
- **Inicialização dos Pesos**: Valores aleatórios entre -1 e 1
- **Função de Ativação**: Sigmoid em todas as camadas

## 📚 Conceitos Utilizados

1. **Forward Propagation**: Propagação dos sinais da entrada até a saída
2. **Backward Propagation**: Propagação do erro da saída até a entrada
3. **Gradient Descent**: Atualização dos pesos minimizando o erro
4. **Derivada da Sigmoid**: `s'(x) = s(x) * (1 - s(x))`

## 🎯 Por que XOR é Importante?

O XOR é um problema não-linear que não pode ser resolvido por um perceptron de camada única. Este problema foi fundamental no desenvolvimento das redes neurais multicamadas e do algoritmo backpropagation, demonstrando a necessidade de camadas ocultas para resolver problemas complexos.

## 📝 Estrutura do Código

```python
class MLP:
    - train(): Treina a rede usando backpropagation
    - predict(): Faz predições após o treinamento
```

## 📈 Resultados

A rede aprende corretamente a função XOR após 50.000 épocas de treinamento, demonstrando a capacidade de uma MLP com camadas ocultas de resolver problemas não-linearmente separáveis.

