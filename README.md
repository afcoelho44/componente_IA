# componente_IA — correção da distância do GPS

API Flask usada pelas dicas quente/frio do project_LocAR. O modelo (Random Forest)
aprende o erro da distância do GPS até o objeto com as distâncias medidas com fita
no experimento, e devolve a distância corrigida. A regra quente/frio fica no app.

## Treinar
```bash
# dados reais: erro_distancia_por_leitura.csv gerado pelo analisar.py
docker compose run --rm api python treinar.py dados/erro_distancia_por_leitura.csv modelos/modelo.joblib --tabela dados/validacao.csv

# sem dados ainda: CSV falso só para testar o caminho todo
docker compose run --rm api python gerar_falso.py dados/falso.csv
docker compose run --rm api python treinar.py dados/falso.csv modelos/modelo.joblib
```
A validação compara GPS bruto × viés médio × Random Forest (RMSE por ambiente,
separando por participante). Essa tabela vai para o trabalho.

## Rodar
```bash
docker compose up --build -d     # reinicie depois de treinar: docker compose restart
curl localhost:5000/saude
curl -X POST localhost:5000/corrigir -H "Content-Type: application/json" \
     -d '{"d_gps_m": 22, "accuracy": 5, "ambiente": "outdoor", "dispositivo": "Android"}'
```
Sem modelo treinado, `/corrigir` responde 503 e o Laravel usa a distância do GPS.

No `.env` do project_LocAR: `IA_URL=http://host.docker.internal:5000` e `php artisan config:clear`.

## Testes
```bash
pip install -r requirements.txt && pytest -q
```

`legado/rotular.py`: rotulagem da ideia antiga (prever quente/frio direto pelos sensores).
