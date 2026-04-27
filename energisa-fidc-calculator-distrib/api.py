"""FastAPI service for the FIDC calculator.

This module exposes the calculation engine.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from v2 import engine as eng


app = FastAPI(
    title="FIDC Calculator",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_upload(upload: UploadFile) -> io.BytesIO:
    content = upload.file.read()
    buffer = io.BytesIO(content)
    buffer.name = upload.filename or "upload.xlsx"
    buffer.seek(0)
    return buffer


def _normalize_data_base(value: str | None) -> str:
    if not value:
        return datetime.utcnow().strftime("%Y-%m-%d")

    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"data_base inválida: {value}")
    return parsed.strftime("%Y-%m-%d")


def _coerce_table(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    frame = df.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    lowered = {column.lower().strip(): column for column in frame.columns}

    if kind == "index":
        if {"data", "indice"}.issubset(lowered):
            frame = frame.rename(columns={lowered["data"]: "data", lowered["indice"]: "indice"})
        elif {"date", "value"}.issubset(lowered):
            frame = frame.rename(columns={lowered["date"]: "data", lowered["value"]: "indice"})
        elif {"ano", "mes", "indice"}.issubset(lowered):
            frame = frame.rename(
                columns={
                    lowered["ano"]: "ano",
                    lowered["mes"]: "mes",
                    lowered["indice"]: "indice",
                }
            )
        elif len(frame.columns) >= 2:
            frame = frame.rename(columns={frame.columns[0]: "data", frame.columns[1]: "indice"})
        else:
            raise ValueError("Arquivo de índices precisa ter pelo menos 2 colunas")

        if {"ano", "mes"}.issubset(frame.columns):
            frame["ano"] = pd.to_numeric(frame["ano"], errors="coerce")
            frame["mes"] = pd.to_numeric(frame["mes"], errors="coerce")
            frame = frame.dropna(subset=["ano", "mes"])
            frame["data"] = pd.to_datetime(
                frame["ano"].astype(int).astype(str)
                + "-"
                + frame["mes"].astype(int).astype(str).str.zfill(2)
                + "-01"
            )

        frame["data"] = pd.to_datetime(frame["data"], errors="coerce")
        frame["indice"] = pd.to_numeric(
            frame["indice"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )
        frame = frame.dropna(subset=["data", "indice"])
        return frame[["data", "indice"]].sort_values("data").reset_index(drop=True)

    if kind == "recovery":
        rename_map: dict[str, str] = {}
        for column in frame.columns:
            normalized = column.lower().strip()
            if normalized in ("empresa", "distribuidora", "cedente"):
                rename_map[column] = "Empresa"
            elif normalized in ("tipo", "tipo titulo", "tipo_titulo"):
                rename_map[column] = "Tipo"
            elif normalized in ("aging", "faixa", "faixa aging"):
                rename_map[column] = "Aging"
            elif "taxa" in normalized and "recuper" in normalized:
                rename_map[column] = "Taxa de recuperação"
            elif "prazo" in normalized or "recebimento" in normalized:
                rename_map[column] = "Prazo de recebimento"

        frame = frame.rename(columns=rename_map)
        required = {"Empresa", "Aging", "Taxa de recuperação", "Prazo de recebimento"}
        if not required.issubset(frame.columns):
            missing = ", ".join(sorted(required - set(frame.columns)))
            raise ValueError(f"Colunas ausentes na taxa de recuperação: {missing}")

        if "Tipo" not in frame.columns:
            frame["Tipo"] = ""

        frame["Taxa de recuperação"] = pd.to_numeric(
            frame["Taxa de recuperação"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        ).fillna(0.0)
        frame["Prazo de recebimento"] = pd.to_numeric(
            frame["Prazo de recebimento"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        ).fillna(6).astype(int)

        return frame[["Empresa", "Tipo", "Aging", "Taxa de recuperação", "Prazo de recebimento"]]

    if kind == "di_pre":
        rename_map: dict[str, str] = {}
        for column in frame.columns:
            normalized = column.lower().strip()
            if normalized in ("meses_futuros", "meses futuros", "prazo", "meses"):
                rename_map[column] = "meses_futuros"
            elif normalized in ("252", "taxa_252", "taxa 252", "di_pre", "di pre", "taxa"):
                rename_map[column] = "252"

        frame = frame.rename(columns=rename_map)
        if {"meses_futuros", "252"}.issubset(frame.columns):
            frame["meses_futuros"] = pd.to_numeric(
                frame["meses_futuros"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
            ).fillna(0).astype(int)
            frame["252"] = pd.to_numeric(
                frame["252"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
            ).fillna(0.0)
            return frame[["meses_futuros", "252"]].dropna()

        raise ValueError("Arquivo DI-PRE precisa ter colunas equivalentes a meses_futuros e 252")

    return frame


def _standardize_base(df: pd.DataFrame, source_name: str, data_base: str, is_voltz_global: bool) -> pd.DataFrame:
    source_name_upper = source_name.upper()
    mapping = eng.auto_detect_columns(list(df.columns))

    standardized = pd.DataFrame(index=df.index)
    for internal_name, source_column in mapping.items():
        standardized[internal_name] = df[source_column]

    if "valor_principal" not in standardized.columns and "valor_principal" in df.columns:
        standardized["valor_principal"] = df["valor_principal"]

    if "data_vencimento" not in standardized.columns and "data_vencimento" in df.columns:
        standardized["data_vencimento"] = df["data_vencimento"]

    if "empresa" not in standardized.columns:
        standardized["empresa"] = "VOLTZ" if (is_voltz_global or "VOLTZ" in source_name_upper) else "DESCONHECIDA"

    for column_name, default_value in {
        "tipo": "",
        "status_conta": "",
        "situacao": "",
        "nome_cliente": "",
        "documento": "",
        "contrato": "",
        "classe": "",
        "valor_nao_cedido": 0,
        "valor_terceiro": 0,
        "valor_cip": 0,
    }.items():
        if column_name not in standardized.columns:
            standardized[column_name] = default_value

    standardized["data_base"] = data_base
    standardized["is_voltz"] = is_voltz_global or "VOLTZ" in source_name_upper

    missing = [column for column in eng.REQUIRED_COLS if column not in standardized.columns]
    if missing:
        raise ValueError(f"Campos obrigatórios ausentes: {', '.join(missing)}")

    return standardized


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "FIDC Calculator API",
        "status": "ok",
        "endpoints": ["/health", "/preview", "/calculate"],
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/preview")
async def preview(data_file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        df = eng.read_uploaded_file(_read_upload(data_file))
        return {
            "filename": data_file.filename,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "detected_mapping": eng.auto_detect_columns(list(df.columns)),
            "preview": df.head(5).to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/calculate")
async def calculate(
    data_file: UploadFile = File(...),
    index_file: UploadFile | None = File(default=None),
    recovery_file: UploadFile | None = File(default=None),
    di_pre_file: UploadFile | None = File(default=None),
    di_pre_taxa_manual: float = Form(default=12.0),
    di_pre_prazo_manual: int = Form(default=6),
    data_base: str | None = Form(default=None),
    spread_percent: float = Form(default=0.025),
    prazo_horizonte: int = Form(default=6),
    is_voltz_global: bool = Form(default=False),
    output: str = Form(default="json"),
) -> Any:
    try:
        resolved_data_base = _normalize_data_base(data_base)
        df_base = eng.read_uploaded_file(_read_upload(data_file))
        df_base = _standardize_base(
            df_base,
            source_name=data_file.filename or "base.xlsx",
            data_base=resolved_data_base,
            is_voltz_global=is_voltz_global,
        )

        idx_df = eng.build_index_series()
        if index_file is not None:
            loaded_idx = eng.load_indices_from_excel(_read_upload(index_file))
            if loaded_idx is not None and not loaded_idx.empty:
                idx_df = loaded_idx

        taxa_df = None
        if recovery_file is not None:
            taxa_df = eng.load_recovery_rates_from_excel(_read_upload(recovery_file))

        di_df = None
        if di_pre_file is not None:
            di_df = eng.load_di_pre_from_excel(_read_upload(di_pre_file))
        else:
            di_df = pd.DataFrame({
            "meses_futuros": list(range(1, di_pre_prazo_manual + 1)),
            "252": [di_pre_taxa_manual] * di_pre_prazo_manual,
        })

        result_df = eng.calculate(
            df=df_base,
            idx_df=idx_df,
            df_taxa=taxa_df,
            df_di_pre=di_df,
            data_base=resolved_data_base,
            spread_percent=spread_percent,
            prazo_horizonte=prazo_horizonte,
            is_voltz_global=is_voltz_global,
        )
        summary = eng.compute_summary(result_df)

        normalized_output = output.lower().strip()
        if normalized_output == "csv":
            csv_bytes = eng.to_csv_bytes(result_df)
            return StreamingResponse(
                io.BytesIO(csv_bytes),
                media_type="text/csv",
                headers={"Content-Disposition": 'attachment; filename="fidc_resultado.csv"'},
            )

        if normalized_output == "excel":
            xlsx_bytes = eng.to_excel_bytes(result_df, summary)
            return StreamingResponse(
                io.BytesIO(xlsx_bytes),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": 'attachment; filename="fidc_resultado.xlsx"'},
            )

        return {
            "status": "ok",
            "filename": data_file.filename,
            "rows": int(len(result_df)),
            "summary": summary,
            "preview": result_df.head(50).to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc