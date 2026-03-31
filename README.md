# titanic_regressao_logistica
# 🚢 CASO PRÁTICO Master D - Titanic: Modelo de Classificação com Regressão Logística

## 📊 Descrição do Projeto
Este projeto tem como objetivo prever a sobrevivência dos passageiros do Titanic utilizando um modelo de **regressão logística**.  

O projeto aborda todo o processo de análise e modelação, incluindo:

- Pré-processamento de dados (tratamento de valores nulos, transformação de variáveis categóricas)
- Treino de modelo de regressão logística
- Avaliação de desempenho usando **matriz de confusão** e **taxa de acerto**

O caso prático foi desenvolvido no contexto do curso Master D de **Python para Análise de Dados**.

---

## 📁 Dataset
O dataset utilizado é o `titanic_v2.csv`, que contém informações sobre os passageiros, incluindo:

- `survived` → Sobrevivência do passageiro (0 = Não, 1 = Sim)  
- `pclass` → Classe do bilhete (1, 2 ou 3)  
- `sex` → Sexo do passageiro  
- `age` → Idade  
- `fare` → Tarifa paga  

Fonte: Titanic Dataset (versão académica)

---

## ⚙️ Etapas do Projeto
1. **Importação das bibliotecas**: Pandas, NumPy, Statsmodels, Scikit-learn  
2. **Carregamento do dataset**: leitura do arquivo `titanic_v2.csv`  
3. **Pré-processamento dos dados**:  
   - Remoção de valores nulos  
   - Transformação de variáveis categóricas (`sex`, `pclass`) em numéricas  
4. **Divisão treino/teste**: 70% treino / 30% teste  
5. **Treino do modelo de regressão logística**  
6. **Avaliação do modelo**: matriz de confusão e taxa de acerto

---

## 🛠️ Tecnologias Utilizadas
- Python 3.x  
- Pandas  
- NumPy  
- Statsmodels  
- Scikit-learn

---

## ▶️ Como Executar o Projeto
1. Clonar o repositório:
```bash
git clone https://github.com/vanilsonManuel/titanic_regressao_logistica.git
