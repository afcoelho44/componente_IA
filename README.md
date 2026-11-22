# componente_IA — classificação da proximidade (quente / morno / frio)

API Flask usada pelas dicas da tela de RA do project_LocAR. Um Random Forest classifica
a proximidade do aluno ao objeto a partir das últimas leituras do GPS:

| Classe | Distância até o objeto |
|---|---|
| quente | ≤ 10 m |
| morno  | > 10 m e ≤ 15 m |
| frio   | > 15 m |

Os limites ficam em `modelo_classe.py` (`LIMITES`). No treino, a classe verdadeira vem da
distância medida com fita no experimento; as features vêm só do GPS (presente e passado):
distância, variação da distância, mediana e desvio das últimas 5 leituras, accuracy, speed
e dispositivo. O treino compara o modelo com a **regra** (os mesmos limites aplicados direto
na distância do GPS).

## Treinar
```bash
# dados reais: leituras_processadas.csv gerado pelo experimento/analisar.py, copiado para dados/
docker compose run --rm api python treinar.py dados/leituras_processadas.csv modelos/modelo.joblib \
    --excluir-local "Entre o bloco F e o RU" --saida-validacao dados/validacao
docker compose restart api

# sem dados: CSV falso só para testar o caminho todo
docker compose run --rm api python gerar_falso.py dados/falso.csv
docker compose run --rm api python treinar.py dados/falso.csv modelos/modelo.joblib
```
- `--ambiente` (padrão `outdoor`) filtra o ambiente; `--excluir-local` tira um ponto (repetível).
- A validação deixa um participante de fora (ou um ponto, se só houver 1 participante) e
  imprime acurácia, F1 macro, matriz de confusão, precisão/recall e trocas de classe por
  captura, para a regra e para o Random Forest. Com `--saida-validacao`, salva as tabelas
  (`resumo.csv`, `por_grupo.csv`, `matriz_regra.csv`, `matriz_rf.csv`, `previsoes.csv`).

## Rodar
```bash
docker compose up --build -d
curl localhost:5000/saude
curl -X POST localhost:5000/classificar -H "Content-Type: application/json" -d '{
  "objeto": {"lat": -26.2535, "lng": -48.8555},
  "leituras": [{"lat": -26.2536, "lng": -48.8555, "accuracy": 4, "speed": 0, "t_ms": 0},
               {"lat": -26.2537, "lng": -48.8555, "accuracy": 4, "speed": 0, "t_ms": 1000}],
  "dispositivo": "Android"}'
# -> {"classe": "quente", "probabilidades": {...}, "dist_m": 22.3, "fonte": "ia"}
```
- `leituras`: as últimas leituras do GPS, da mais antiga para a mais recente (até 20). A classe é da última.
- Se a distância passar de 1,5 × a maior distância vista no treino, responde pela regra (`fonte: "regra"`).
- Sem modelo treinado (ou treinado com outros limites/features), `/classificar` responde 503 e o Laravel usa a regra.

No `.env` do project_LocAR: `IA_URL=http://host.docker.internal:5000` e `php artisan config:clear`.

## Testes
```bash
pip install -r requirements.txt && pytest -q
```

## Legado
- `legado/correcao_distancia/`: versão anterior (modelo que corrigia a distância do GPS, rota `/corrigir`).
  Abandonada: o erro que sobra é um desvio fixo do GPS por captura, que não dá para corrigir na aplicação.
- `legado/rotular.py`: primeira ideia (prever quente/frio pela tendência, com os sensores).
