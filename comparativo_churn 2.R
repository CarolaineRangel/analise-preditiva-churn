# Carregar pacotes
pacotes <- c("tidyverse", "rpart", "rpart.plot", "readxl", "caTools", "fastDummies", "caret",
             "pROC", "plotly", "forcats", "randomForest", "gbm", "xgboost", "ROSE","scales","naivebayes")


# Verificar e instalar pacotes ausentes
pacotes_instalar <- pacotes[!sapply(pacotes, requireNamespace, quietly = TRUE)]
if (length(pacotes_instalar) > 0) {
  install.packages(pacotes_instalar, dependencies = TRUE)
}

# Carregar pacotes
lapply(pacotes, library, character.only = TRUE)

library(caret)
library(ROSE)
library(ggplot2)
library(dplyr)
library(forcats)
library(readxl)
library(rpart)
library(rpart.plot)
library(pROC)

# Carregar os dados
clientes <- read_excel("base_de_clientes_1.xlsx")

# Renomear colunas
clientes <- rename(clientes, 
                   ativo = "Fase do ciclo de vida",
                   vendas_30_dias = "Vendas Últimos 30 dias",
                   pedidos_30_dias = "Pedidos Últimos 30 dias",
                   modelo_negocio = "Modelo de Negócio",
                   forma_pagamento = "Forma de pagamento",
                   plano_escolhido = "Plano Escolhido",
                   desconto = "Desconto ofertado",
                   mensalidade = "Valor da mensalidade",
                   canal = "Atribuição de Canal",
                   plano_pagamento = "Plano de Pagamento",
                   pedidos_delivery = "Pedidos Últimos 30 dias Delivery",
                   vendas_delivery = "Vendas Últimos 30 dias Delivery",
                   franquia ="Franquia?",
                   dias_boleto_vencido = "Dias de boleto vencido")


# Transformar coluna ativo em binária, onde 0 é inativo e 1 é cliente
clientes$ativo <- ifelse(clientes$ativo == "Cliente", 1, 0)

# Lista das variáveis a serem transformadas em fatores
categorical_vars <- c("ativo", "modelo_negocio", "forma_pagamento", "plano_escolhido", 
                      "desconto", "franquia", "UF", "canal", "plano_pagamento","pedidos_30_dias")

# Transformar as variáveis especificadas em fatores
clientes[categorical_vars] <- lapply(clientes[categorical_vars], factor)

# Verificar o resultado
str(clientes)

# Verificar se a variável resposta tem mais de uma classe, sendo 1 cliente e 0 não cliente
table(clientes$ativo)

################################################################################
######################### Árvores de Decisão treino ############################
################################################################################

# Separação da base de dados em bases de treino e teste
set.seed(100)
#library(caret)

# Criar índices estratificados para divisão de treino/teste
index <- createDataPartition(clientes$ativo, p = 0.70, list = FALSE)
treino <- clientes[index, ]
teste <- clientes[-index, ]


# Gerando a árvore de decisão (árvore de classificação)
set.seed(100)
arvore <- rpart(formula = ativo ~ .,
                data = treino,
                parms = list(split = "gini"),
                method = "class",
                control = rpart.control(minsplit = 8,
                                        maxdepth = 5,
                                        minbucket = 5,
                                        cp = 0.0028))

paleta = scales::viridis_pal(begin=.30, end=1)(10)
#Plot fancy
rpart.plot::rpart.plot(arvore,
                       box.palette = paleta,
                       extra = 100,
                       branch.lty=5,
                       shadow.col="gray",
                       nn = TRUE) # Paleta de cores
# Obter o número total de nós (incluindo nós internos e folhas)
nnodes <- sum(arvore$where != 0)
################################################################################
######################### Avaliação do Modelo treino ###########################
################################################################################

# Previsões no conjunto de treino
preditos_treino <- predict(arvore, treino, type = "prob")

# Definir cutoff para classificação binária (ativa/inativa)
cutoff <- 0.5

# Classificando os valores preditos com base no cutoff
preditos_class <- as.factor(ifelse(preditos_treino[, "1"] >= cutoff, "1", "0"))
table(preditos_class)

library(caret)
# Matriz de confusão para o cutoff estabelecido
conf_matrix_treino <- confusionMatrix(data = preditos_class,
                reference = treino$ativo, 
                positive = "1")

# Extração de precisão e recall da matriz de confusão
precision_treino <- conf_matrix_treino$byClass["Pos Pred Value"]  # Precisão
recall_treino <- conf_matrix_treino$byClass["Sensitivity"]        # Recall

# Imprimindo os resultados de precisão e recall
print(paste("Precisão:", precision_treino))
print(paste("Recall:", recall_treino))

# Resultados relacionados ao parâmetro de complexidade
library(rpart)

plotcp(arvore)
printcp(arvore)

# chamando novamente as biblotecas
library(ggplot2)
library(pROC)

# Calcular a curva ROC
valor_auc_treino_arvore <- roc(response = as.numeric(treino$ativo),  # Convertendo para binário: "1" -> 1, "0" -> 0
           predictor = preditos_treino[, "1"])

# Criar um data frame para plotagem da curva ROC
roc_data <- data.frame(
  specificity = 1 - valor_auc_treino_arvore$specificities,
  sensitivity = valor_auc_treino_arvore$sensitivities
)

# Plot da curva ROC
plotROC <- ggplot(data = roc_data, aes(x = specificity, y = sensitivity)) +
  geom_line(color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  labs(x = "1 - Especificidade",
       y = "Sensitividade",
       title = paste("Curva ROC Treino AD (AUC =", round(valor_auc_treino_arvore$auc, 3), ")")) +
  theme_bw()

# Exibir o plot
print(plotROC)


library(tibble)
library(dplyr)   # Para mutate, arrange, rename
library(forcats)  # Para fct_inorder

# Calcular a importância das variáveis
imp_arvore <- data.frame(importancia = arvore$variable.importance) %>% 
  rownames_to_column() %>% 
  arrange(importancia) %>% 
  rename(variavel = rowname) %>%
  mutate(variavel = fct_inorder(variavel))

# Plotar a importância das variáveis
ggplot(imp_arvore) +
  geom_segment(aes(x = variavel, y = 0, xend = variavel, yend = importancia), 
               linewidth = 1.5, alpha = 0.7) +  # Substituir 'size' por 'linewidth'
  geom_point(aes(x = variavel, y = importancia, col = variavel), 
             size = 4, show.legend = FALSE) +
  coord_flip() +
  labs(x = "Variável", y = "Importância") +
  theme_bw()


################################################################################
######################### Investigando se há overfitting #######################
################################################################################

# Valores preditos pela árvore (base de teste)
preditos_teste <- predict(arvore, teste)

# Classificando os valores preditos com base no cutoff
preditos_class_teste = factor(ifelse(preditos_teste[,2] > cutoff, 1, 0))

# Matriz de confusão para o cutoff estabelecido na base de teste
conf_matrix_teste <-confusionMatrix(data = preditos_class_teste,
                               reference = teste$ativo, 
                               positive = "1")
# Extração de precisão e recall da matriz de confusão
precision_teste <- conf_matrix_teste$byClass["Pos Pred Value"]  # Precisão
recall_teste <- conf_matrix_teste$byClass["Sensitivity"]        # Recall

# Imprimindo os resultados de precisão e recall
print(paste("Precisão:", precision_teste))
print(paste("Recall:", recall_teste))
print(conf_matrix_teste)
print (conf_matrix_treino)
# Função para plotar matriz de confusão
plot_confusion <- function(conf_matrix, title) {
  conf_df <- as.data.frame(conf_matrix$table)
  conf_df$Reference <- factor(conf_df$Reference, levels = c("0", "1"))
  conf_df$Prediction <- factor(conf_df$Prediction, levels = c("0", "1"))
  
  ggplot(data = conf_df, aes(x = Reference, y = Prediction, fill = as.factor(Freq))) +
    geom_tile() +
    geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
    labs(x = "Valor Real", y = "Valor Previsto", fill = "Contagem", title = title) +
    scale_fill_gradient(low = "white", high = "blue", na.value = "grey50", aesthetics = "colour") +  # Ajuste na escala de cores
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Plot da matriz de confusão para o conjunto de treino e teste
plot_confusion(conf_matrix_treino, "Matriz de Confusão - Treino AD")
plot_confusion(conf_matrix_teste, "Matriz de Confusão - Teste AD")


# Calcular a curva ROC
valor_auc_teste_arvore <- roc(response = as.numeric(teste$ativo),  # Convertendo para binário: "1" -> 1, "0" -> 0
                               predictor = preditos_teste[, "1"])

# Criar um data frame para plotagem da curva ROC
roc_data <- data.frame(
  specificity = 1 - valor_auc_teste_arvore$specificities,
  sensitivity = valor_auc_teste_arvore$sensitivities
)

# Plot da curva ROC
plotROC <- ggplot(data = roc_data, aes(x = specificity, y = sensitivity)) +
  geom_line(color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  labs(x = "1 - Especificidade",
       y = "Sensitividade",
       title = paste("Curva ROC Teste AD (AUC =", round(valor_auc_teste_arvore$auc, 3), ")")) +
  theme_bw()

# Exibir o plot
print(plotROC)
printcp(arvore)
################################################################################
######################### Ajuste de Parâmetros #################################
###############################################################################

# Grid Search
## Procedimento que estima vários modelos alterando os hiperparâmetros
## O intuito é identificar quais hiperparâmetros geram o "melhor" modelo

# Parametrizando o grid
grid <- expand.grid(minsplit = seq(from = 1, to = 10, by = 2),
                    maxdepth = seq(from = 1, to = 10, by = 2),
                    minbucket = seq(from = 1, to = 10, by = 2),
                    cp = c(0.00001,0.0001, 0.001,0.005, 0.01))
# Criando uma lista para armazenar os resultados
modelos <- list()

# Gerando um processo iterativo
for (i in 1:nrow(grid)) {
  # Coletando os parâmetros do grid
  minsplit <- grid$minsplit[i]
  maxdepth <- grid$maxdepth[i]
  minbucket <- grid$minbucket[i]
  cp <- grid$cp[i]
  
  # Estimando os modelos e armazenando os resultados
  set.seed(100)
  modelos[[i]] <- rpart(
    formula = ativo ~ .,
    data = treino,
    parms = list(split = 'gini'),
    method = "class",
    control = rpart.control(
      minsplit = minsplit, 
      maxdepth = maxdepth,
      minbucket = minbucket, 
      cp = cp
    )
  )
  }
# Função para coletar parâmetro cp dos modelos
coleta_cp <- function(x) {
  min <- which.min(x$cptable[, "xerror"])
  cp <- x$cptable[min, "CP"] 
}

# Função para coletar o erro mínimo dos modelos
coleta_erro <- function(x) {
  min <- which.min(x$cptable[, "xerror"])
  xerror <- x$cptable[min, "xerror"] 
}

# Analisando os resultados
grid %>%
  mutate(
    cp = purrr::map_dbl(modelos, coleta_cp),
    erro = purrr::map_dbl(modelos, coleta_erro)
  ) %>%
  arrange(erro) %>% 
  slice_head(n = 10)

# Modelo finalF1-score
set.seed(100)
arvore_grid <- rpart(formula = ativo ~ .,
                data = treino,
                parms = list(split = "gini"),
                method = "class",
                control = rpart.control(minsplit = 1,
                                        maxdepth = 9,
                                        minbucket = 9,
                                        cp = 0.0015625))


# Valores preditos pela árvore (base de treino)
preditos_grid_treino <- predict(arvore_grid, treino, type = "prob")

# Classificando os valores preditos com base no cutoff
preditos_class_grid_treino = factor(ifelse(preditos_grid_treino[,2] > cutoff, 1, 0))

# Matriz de confusão para o cutoff estabelecido na base de teste
confMatrixTreino <-confusionMatrix(data = preditos_class_grid_treino,
                                  reference = treino$ativo, 
                                  positive = "1")
print(confMatrixTreino)

# Valores preditos pela árvore (base de teste)
preditos_grid_teste <- predict(arvore_grid, teste, type = "prob")

# Classificando os valores preditos com base no cutoff
preditos_class_grid_teste = factor(ifelse(preditos_grid_teste[,2] > cutoff, 1, 0))

# Matriz de confusão para o cutoff estabelecido na base de teste
confMatrixTeste <-confusionMatrix(data = preditos_class_grid_teste,
                                    reference = teste$ativo, 
                                    positive = "1")

print(confMatrixTeste)

# Função para plotar matriz de confusão
plot_confusion <- function(confusionMatrix, title) {
  conf_df <- as.data.frame(confusionMatrix$table)
  conf_df$Reference <- factor(conf_df$Reference, levels = c("0", "1"))
  conf_df$Prediction <- factor(conf_df$Prediction, levels = c("0", "1"))
  
  ggplot(data = conf_df, aes(x = Reference, y = Prediction, fill = as.factor(Freq))) +
    geom_tile() +
    geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
    labs(x = "Valor Real", y = "Valor Previsto", fill = "Contagem", title = title) +
    scale_fill_gradient(low = "white", high = "blue", na.value = "grey50", aesthetics = "colour") +  # Ajuste na escala de cores
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  }
# Plot da matriz de confusão para o conjunto de treino e teste
plot_confusion(confMatrixTreino, "Matriz de Confusão - Treino AD")
plot_confusion(confMatrixTeste, "Matriz de Confusão - Teste AD")


# Extração de precisão e recall da matriz de confusão
preditos_grid_treino <- confMatrixTreino$byClass["Pos Pred Value"]  # Precisão
recall_treino <- confMatrixTreino$byClass["Sensitivity"]        # Recall

# Imprimindo os resultados de precisão e recall
print(paste("Precisão:", preditos_grid_treino))
print(paste("Recall:", recall_treino))

# Extração de precisão e recall da matriz de confusão
precision_teste <- confMatrixTeste$byClass["Pos Pred Value"]  # Precisão
recall_teste <- confMatrixTeste$byClass["Sensitivity"]        # Recall

# Imprimindo os resultados de precisão e recall
print(paste("Precisão:", precision_teste))
print(paste("Recall:", recall_teste))

preditos_grid_treino  <- predict(arvore_grid, treino, type = "prob")
# Calcular a curva ROC
valor_auc_treino_arvore_final <- roc(response = as.numeric(treino$ativo),  # Convertendo para binário: "1" -> 1, "0" -> 0
                              predictor = preditos_grid_treino[, "1"])

# Criar um data frame para plotagem da curva ROC
roc_data <- data.frame(
  specificity = 1 - valor_auc_treino_arvore_final$specificities,
  sensitivity = valor_auc_treino_arvore_final$sensitivities
)

# Plot da curva ROC
plotROC <- ggplot(data = roc_data, aes(x = specificity, y = sensitivity)) +
  geom_line(color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  labs(x = "1 - Especificidade",
       y = "Sensitividade",
       title = paste("Curva ROC Teste AD (AUC =", round(valor_auc_treino_arvore_final$auc, 3), ")")) +
  theme_bw()
# Exibir o plot
print(plotROC)
# -----------------------------------------------------------------------------------------

# Calcular a curva ROC teste
valor_auc_teste_arvore_final <- roc(response = as.numeric(teste$ativo),  # Convertendo para binário: "1" -> 1, "0" -> 0
                                     predictor = preditos_grid_teste[, "1"])

# Criar um data frame para plotagem da curva ROC
roc_data <- data.frame(
  specificity = 1 - valor_auc_teste_arvore_final$specificities,
  sensitivity = valor_auc_teste_arvore_final$sensitivities
)

# Plot da curva ROC
plotROC <- ggplot(data = roc_data, aes(x = specificity, y = sensitivity)) +
  geom_line(color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  labs(x = "1 - Especificidade",
       y = "Sensitividade",
       title = paste("Curva ROC Teste AD (AUC =", round(valor_auc_teste_arvore_final$auc, 3), ")")) +
  theme_bw()
# Exibir o plot
print(plotROC)

################################################################################
######################### Naive Bayes ###########################################
################################################################################

library(naivebayes)

# Ajuste do modelo Naive Bayes
nb_model <- naive_bayes(ativo ~ ., data = treino, laplace = 1)

# Verificar as variáveis usadas no modelo
vars_usadas <- names(nb_model$tables)

# Verificar as variáveis usadas
print(vars_usadas)

# Selecionar apenas as variáveis usadas no conjunto de dados de treino
treino_restrito <- treino[, vars_usadas]

# Verificar se as variáveis estão corretas
print(colnames(treino_restrito))

# Prever com o modelo Naive Bayes usando apenas as variáveis relevantes
preditos_treino_nb <- predict(nb_model, newdata = treino_restrito, type = "class")

# Ajustar os níveis da variável de referência
treino$ativo <- factor(treino$ativo, levels = levels(preditos_treino_nb))

# Matriz de confusão para o conjunto de treino
conf_matrix_treino_nb <- confusionMatrix(data = preditos_treino_nb, 
                                         reference = treino$ativo, 
                                         positive = "1")
print(conf_matrix_treino_nb)
# Extração de precisão e recall da matriz de confusão
precision_treino_nb <- conf_matrix_treino_nb$byClass["Pos Pred Value"]  # Precisão
recall_treino_nb <- conf_matrix_treino_nb$byClass["Sensitivity"]        # Recall

# Imprimindo os resultados de precisão e recall
print(paste("Naive Bayes - Precisão (treino):", precision_treino_nb))
print(paste("Naive Bayes - Recall (treino):", recall_treino_nb))


# Selecionar apenas as variáveis usadas no conjunto de dados de treino
teste_restrito <- teste[, vars_usadas]

# Verificar se as variáveis estão corretas
print(colnames(teste_restrito))

# Avaliação na base de teste
preditos_teste_nb <- predict(nb_model, newdata = teste_restrito, type = "class")

# Verificar os níveis dos fatores
levels(preditos_teste_nb)
levels(teste$ativo)

# Reordenar os níveis de preditos_teste_nb para corresponder aos níveis de teste$ativo
preditos_teste_nb <- factor(preditos_teste_nb, levels = levels(teste$ativo))

# Matriz de confusão para o conjunto de teste
conf_matrix_teste_nb <- confusionMatrix(data = preditos_teste_nb, 
                                        reference = teste$ativo, 
                                        positive = "1")
print(conf_matrix_teste_nb)
# Extração de precisão e recall da matriz de confusão
precision_teste_nb <- conf_matrix_teste_nb$byClass["Pos Pred Value"]  # Precisão
recall_teste_nb <- conf_matrix_teste_nb$byClass["Sensitivity"]        # Recall

# Imprimindo os resultados de precisão e recall
print(paste("Naive Bayes - Precisão (teste):", precision_teste_nb))
print(paste("Naive Bayes - Recall (teste):", recall_teste_nb))

str(preditos_treino_nb)


#Função para plotar matriz de confusão
plot_confusion <- function(confusionMatrix, title) {
  conf_df <- as.data.frame(confusionMatrix$table)
  conf_df$Reference <- factor(conf_df$Reference, levels = c("0", "1"))
  conf_df$Prediction <- factor(conf_df$Prediction, levels = c("0", "1"))
  
  ggplot(data = conf_df, aes(x = Reference, y = Prediction, fill = as.factor(Freq))) +
    geom_tile() +
    geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
    labs(x = "Valor Real", y = "Valor Previsto", fill = "Contagem", title = title) +
    scale_fill_gradient(low = "white", high = "blue", na.value = "grey50", aesthetics = "colour") +  # Ajuste na escala de cores
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Plot da matriz de confusão para o conjunto de treino e teste
plot_confusion(conf_matrix_treino_nb, "Matriz de Confusão - Treino NB")
plot_confusion(conf_matrix_teste_nb, "Matriz de Confusão - Teste NB")

# Converter preditos_treino_nb de fator para numérico
preditos_treino_nb_numeric <- as.numeric(as.character(preditos_treino_nb))

# Calcular a curva ROC
ROC_nb <- roc(response = as.numeric(treino$ativo),
              predictor = preditos_treino_nb_numeric)

# Calcular a AUC
valor_auc_treino_nb <- auc(ROC_nb)

# Criar um data frame para plotagem da curva ROC
roc_data <- data.frame(
  specificity = 1 - ROC_nb$specificities,
  sensitivity = ROC_nb$sensitivities
)

# Plot da curva ROC
plotROC <- ggplot(data = roc_data, aes(x = specificity, y = sensitivity)) +
  geom_line(color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  labs(x = "1 - Especificidade",
       y = "Sensitividade",
       title = paste("Curva ROC Treino NB (AUC =", round(valor_auc_treino_nb, 3), ")")) +
  theme_bw()

# Exibir o gráfico
print(plotROC)


# Calcular a curva ROC TESTE

# Converter preditos_treino_nb de fator para numérico
preditos_teste_nb_numeric <- as.numeric(as.character(preditos_teste_nb))

# Calcular a curva ROC
ROC_nb <- roc(response = as.numeric(teste$ativo),
              predictor = preditos_teste_nb_numeric)

# Calcular a AUC
valor_auc_teste_nb <- auc(ROC_nb)

# Criar um data frame para plotagem da curva ROC
roc_data <- data.frame(
  specificity = 1 - ROC_nb$specificities,
  sensitivity = ROC_nb$sensitivities
)

# Plot da curva ROC
plotROC <- ggplot(data = roc_data, aes(x = specificity, y = sensitivity)) +
  geom_line(color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  labs(x = "1 - Especificidade",
       y = "Sensitividade",
       title = paste("Curva ROC Teste NB (AUC =", round(valor_auc_teste_nb, 3), ")")) +
  theme_bw()

# Exibir o gráfico
print(plotROC)


################################################################################
######################### Random Forest ########################################
################################################################################

library(caret)
library(randomForest)
treino$vendas_30_dias <- as.integer(as.character(treino$vendas_30_dias))
teste$vendas_30_dias <- as.integer(as.character(teste$vendas_30_dias))
treino$pedidos_30_dias <- as.integer(as.character(treino$pedidos_30_dias))
teste$pedidos_30_dias <- as.integer(as.character(teste$pedidos_30_dias))
# Usar drop_na() para remover linhas com NAs
treino <- drop_na(treino)
teste <- drop_na(teste)

# Verificar se ainda há valores ausentes
sum(is.na(treino))  # Deve retornar 0
sum(is.na(teste))   # Deve retornar 0

# Ajustar o modelo Random Forest
set.seed(100)
random_forest <- randomForest(ativo ~ .,
                              data = treino,
                              importance = TRUE,
                              ntree = 200,
                              mtry = 2)

# Definir a semente para reproducibilidade
set.seed(100)

random_forest <- randomForest(ativo ~.,
                              data = treino,
                              importance=TRUE,
                              ntree = 200,
                              mtry = 2)

# "importance": para retornar a importância das variáveis X no modelo
# "ntree": número de árvores a serem geradas
# "mtry": número de variáveis amostradas aleatoriamente para tentar cada divisão
### o valor de mtry não pode exceder a quantidade de variáveis X

# Dois outros possíveis hiperparâmetros relacionados às árvores:
## "nodesize": tamanho mínimo dos nós folha
## "maxnodes": número máximo de nós folha que a árvore pode ter

# Calcular as previsões de classe para o conjunto de treino
preditos_treino_rf_class <- predict(random_forest, newdata = treino, type = "response")

# Calcular as probabilidades preditas para a classe positiva (ativo = 1)
preditos_treino_rf_prob <- predict(random_forest, newdata = treino, type = "prob")

## Note que não foi estabelecido um cutoff (o algoritmo usou o padrão)

# Matriz de confusão para a amostra de treino
conf_matrix_treino_rf <- confusionMatrix(preditos_treino_rf_class, treino$ativo, positive = "1")

print(conf_matrix_treino_rf)


# Extração de precisão e recall da matriz de confusão
precision_treino_rf <- conf_matrix_treino_rf$byClass["Pos Pred Value"]  # Precisão
recall_treino_rf <- conf_matrix_treino_rf$byClass["Sensitivity"]        # Recall

# Imprimindo os resultados de precisão e recall
print(paste("Precisão:", precision_treino_rf))
print(paste("Recall:", recall_treino_rf))

# Carregar pacotes
library(pROC)
library(ggplot2)
library(plotly)


# Calcular a curva ROC
roc_curve <- roc(response = treino$ativo, 
                 predictor = preditos_treino_rf_prob[, 2])

# AUC da classe positiva (ativo = 1)
valor_auc_treino_rf <- roc_curve$auc

# Imprimir o valor do AUC
print(paste("Área abaixo da curva (AUC):", round(valor_auc_treino_rf, 5)))

# Plot da curva ROC
plot(roc_curve, main = paste("Curva ROC Treino RF (AUC):", round(valor_auc_treino_rf, 5)), col = "red")



# Investigando se há overfitting

# Calcular as previsões de classe para o conjunto de teste
preditos_teste_rf_class <- predict(random_forest, newdata = teste, type = "response")

# Calcular as probabilidades preditas para a classe positiva (ativo = 1)
preditos_teste_rf_prob <- predict(random_forest, newdata = teste, type = "prob")

# Verificar os níveis dos fatores
levels(preditos_teste_rf_class)
levels(teste$ativo)

# Ajustar os níveis dos valores preditos para corresponder aos níveis do conjunto de teste
preditos_teste_rf_class <- factor(preditos_teste_rf_class, levels = levels(teste$ativo))

# Calcular a matriz de confusão para a amostra de teste
conf_matrix_teste_rf <- confusionMatrix(data = preditos_teste_rf_class,
                                        reference = teste$ativo, 
                                        positive = "1")

# Exibir a matriz de confusão e as estatísticas associadas
print(conf_matrix_teste_rf)

# Extração de precisão e recall da matriz de confusão
precision_teste_rf <- conf_matrix_teste_rf$byClass["Pos Pred Value"]  # Precisão
recall_teste_rf <- conf_matrix_teste_rf$byClass["Sensitivity"]        # Recall

# Imprimindo os resultados de precisão e recall
print(paste("Precisão:", precision_teste_rf))
print(paste("Recall:", recall_teste_rf))


# Calcular a curva ROC
roc_curve <- roc(response = teste$ativo, 
                 predictor = preditos_teste_rf_prob[, 2])

# AUC da classe positiva (ativo = 1)
valor_auc_teste_rf <- roc_curve$auc

# Imprimir o valor do AUC
print(paste("Área abaixo da curva (AUC):", round(valor_auc_teste_rf, 5)))

# Plot da curva ROC
plot(roc_curve, main = paste("Curva ROC Teste RF (AUC):", round(valor_auc_teste_rf, 5)), col = "red")


#Função para plotar matriz de confusão
plot_confusion <- function(confusionMatrix, title) {
  conf_df <- as.data.frame(confusionMatrix$table)
  conf_df$Reference <- factor(conf_df$Reference, levels = c("0", "1"))
  conf_df$Prediction <- factor(conf_df$Prediction, levels = c("0", "1"))
  
  ggplot(data = conf_df, aes(x = Reference, y = Prediction, fill = as.factor(Freq))) +
    geom_tile() +
    geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
    labs(x = "Valor Real", y = "Valor Previsto", fill = "Contagem", title = title) +
    scale_fill_gradient(low = "white", high = "blue", na.value = "grey50", aesthetics = "colour") +  # Ajuste na escala de cores
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Plot da matriz de confusão para o conjunto de treino e teste
plot_confusion(conf_matrix_treino_rf, "Matriz de Confusão - Treino RF")
plot_confusion(conf_matrix_teste_rf, "Matriz de Confusão - Teste RF")

################################################################################
#################### Calcular métricas para os diferentes modelos ##############
################################################################################

# Calcular métricas para os diferentes modelos
# Supondo que você já tenha as variáveis necessárias calculadas para cada modelo

# Confusion Matrix
conf_matrix <- c(
  treino_arvore = conf_matrix_treino$overall["Accuracy"],
  teste_arvore = conf_matrix_teste$overall["Accuracy"],
  treino_nb = conf_matrix_treino_nb$overall["Accuracy"],
  teste_nb = conf_matrix_teste_nb$overall["Accuracy"],
  treino_rf = conf_matrix_treino_rf$overall["Accuracy"],
  teste_rf = conf_matrix_teste_rf$overall["Accuracy"]
)
print(conf_matrix)

# AUC
auc <- list(
  treino_arvore = valor_auc_treino_arvore,
  teste_arvore = valor_auc_teste_arvore,
  treino_nb = valor_auc_treino_nb,
  teste_nb = valor_auc_teste_nb,
  treino_rf = valor_auc_treino_rf,
  teste_rf = valor_auc_teste_rf
)

# Precision e Recall
precision <- c(
  treino_arvore = precision_treino,
  teste_arvore = precision_teste,
  treino_nb = precision_treino_nb,
  teste_nb = precision_teste_nb,
  treino_rf = precision_treino_rf,
  teste_rf = precision_teste_rf
)
print(precision)

recall <- c(
  treino_arvore = recall_treino,
  teste_arvore = recall_teste,
  treino_nb = recall_treino_nb,
  teste_nb = recall_teste_nb,
  treino_rf = recall_treino_rf,
  teste_rf = recall_teste_rf
)
print(recall)

# Definindo as métricas com valores de exemplo
precision <- c(0.9341432, 0.8053221 , 0.9313525, 0.9280576, 0.9885787, 0.8501362)
recall <- c(0.9262039 , 0.8394161, 0.5684803, 0.5649635 , 0.9743590, 0.9109489)
AUC <- c(0.969, 0.815, 0.758, 0.755, 0.99582, 0.93415)



# Construção do data frame resultados
resultados <- data.frame(
  Modelo = rep(c("Árvore de Decisão", "Naive Bayes", "Random Forest"), each = 6),  
  Métrica = rep(c("Precisão Treino", "Recall Treino", "AUC Treino", "Precisão Teste", "Recall Teste", "AUC Teste"), times = 3),
  Conjunto = rep(c("Treino", "Teste", "Treino", "Teste", "Treino", "Teste"), times = 3),  
  Valor = c(
    precision[1], recall[1], AUC[1],  
    precision[2], recall[2], AUC[2],  
    precision[3], recall[3], AUC[3],  
    precision[4], recall[4], AUC[4],  
    precision[5], recall[5], AUC[5],  
    precision[6], recall[6], AUC[6]   
  )
)

# Plotar gráfico comparativo
library(ggplot2)
library(dplyr)

ggplot(resultados, aes(x = Modelo, y = Valor, fill = Conjunto)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  facet_grid(. ~ Métrica, scales = "free_y") +  # Permitir escalas livres em y para cada facet
  labs(x = "Modelo", y = "Valor", fill = "Conjunto") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
