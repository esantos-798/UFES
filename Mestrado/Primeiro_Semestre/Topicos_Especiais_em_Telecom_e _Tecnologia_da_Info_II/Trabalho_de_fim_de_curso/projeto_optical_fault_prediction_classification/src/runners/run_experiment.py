from datetime import datetime
import torch
import torch.optim as optim

from src.data.dataloader import get_dataloader
from src.models.model_factory import ModelFactory

from src.training.trainer import Trainer
from src.training.evaluator import Evaluator

from src.evaluation.anomaly_detector import AnomalyDetector
from src.utils.experiment_logger import ExperimentLogger
from src.training.forecast_evaluator import ForecastEvaluator
from src.utils.hybrid_pipeline import HybridClassifierPipeline  # Importando o Pipeline do XGBoost

class ExperimentRunner:

    def __init__(self, experiment):
        self.experiment = experiment
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.logger = ExperimentLogger(experiment)
        
        # Inicializando os atributos na classe para garantir consistência
        self.model = None
        self.history = None
        self.results = None
        self.threshold = None
        self.predictions = None
        self.labels = None
        self.errors = None
        self.execution_time = None

    def run(self):
        # Adiciona tempo de execução (Início)
        start = datetime.now()

        print()
        print("=" * 60)
        print(self.experiment.name)
        print("=" * 60)
        print()

        # ======================================================
        # DATA (Construindo o caminho do arquivo real)
        # ======================================================
        train_loader, val_loader, test_loader = get_dataloader(
            dataset_path=f"datasets/{self.experiment.dataset}_dataset.csv", 
            batch_size=self.experiment.batch_size,
            task=self.experiment.task
        )

        # ======================================================
        # MODEL
        # ======================================================
        model = ModelFactory.create(self.experiment).to(self.device)

        print()
        print(model)
        print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")

        # ======================================================
        # OPTIMIZER
        # ======================================================
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.experiment.lr,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
        
        # ======================================================
        # TRAIN
        # ======================================================
        trainer = Trainer(
            experiment=self.experiment,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=self.experiment.criterion,
            device=self.device,
            scheduler=scheduler
        )

        model = trainer.fit()
        self.model = model

        history = trainer.history
        self.history = history
        
        # ======================================================
        # TEST
        # ======================================================
        if self.experiment.task == "classification":
            evaluator = Evaluator(
                model=model,
                test_loader=test_loader,
                device=self.device,
                task="classification"
            )

            results = evaluator.evaluate()
            predictions = evaluator.predictions
            labels = evaluator.labels
            errors = None
            threshold = None

            # Executa o pipeline híbrido para classificação pura
            hybrid_pipeline = HybridClassifierPipeline(model=model, device=self.device)
            xgb_metrics = hybrid_pipeline.fit_and_evaluate(train_loader, val_loader, test_loader)
            results.update(xgb_metrics)

        elif self.experiment.task == "forecast":
            predictions = None
            labels = None
            errors = None
            threshold = None

            # ======================================================
            # Forecast Evaluation
            # ======================================================
            print()
            print("=" * 60)
            print("FORECAST EVALUATION")
            print("=" * 60)

            forecast_eval = ForecastEvaluator(
                model=model,
                test_loader=test_loader,
                device=self.device,
                experiment=self.experiment
            )

            forecast_metrics = forecast_eval.evaluate()

            for k, v in forecast_metrics.items():
                print(f"{k}: {v}")

            # ======================================================
            # Anomaly Detection
            # ======================================================
            print()
            print("=" * 60)
            print("ANOMALY DETECTION")
            print("=" * 60)

            current_alpha = getattr(self.experiment, 'alpha', 0.5)

            # Instanciação alinhada com a ordem real dos parâmetros em AnomalyDetector
            detector = AnomalyDetector(
                model=model,
                val_loader=val_loader,
                test_loader=test_loader,
                device=self.device,
                alpha=current_alpha
            )

            detection_metrics = detector.evaluate()

            predictions = detector.predictions
            labels = detector.labels
            errors = detector.errors
            threshold = detector.threshold

            # Unifica as métricas de previsão e detecção
            results = {
                **forecast_metrics,
                **detection_metrics
            }

            # Garante aliases e mapeamentos das métricas operacionais para o JSON final
            results.update({
                "FAR": detection_metrics.get("False Alarm Rate", 0.0),
                "MDR": detection_metrics.get("Miss Detection Rate", 0.0),
                "Average Lead Time": detection_metrics.get("Average Lead Time", 0.0),
                "lead_time_avg": detection_metrics.get("Average Lead Time", 0.0),
                "Min Lead Time": detection_metrics.get("Minimum Lead Time", 0.0),
                "lead_time_min": detection_metrics.get("Minimum Lead Time", 0.0),
                "Max Lead Time": detection_metrics.get("Maximum Lead Time", 0.0),
                "lead_time_max": detection_metrics.get("Maximum Lead Time", 0.0)
            })

            # ======================================================
            # MÓDULO HÍBRIDO: Extração de Features + XGBoost
            # ======================================================
            #try:
            #    if hasattr(model, "is_hybrid") and model.is_hybrid:
            #        xgb_metrics = model.fit_xgboost_and_shap(
            #            train_loader=train_loader,
            #            val_loader=val_loader,
            #            test_loader=test_loader,
            #            device=self.device,
            #            output_dir=self.experiment.output_dir # Salva na pasta do Run
            #        )
            #        results.update(xgb_metrics)
            #    else:
            #        hybrid_pipeline = HybridClassifierPipeline(model=model, device=self.device)
            #        xgb_metrics = hybrid_pipeline.fit_and_evaluate(train_loader, val_loader, test_loader)
            #        results.update(xgb_metrics)
            #        
            #except Exception as e:
            #    print(f"\n[Aviso Híbrido] Ignorando XGBoost para este modelo: {e}")

        else:
            raise ValueError("Unknown task")

        self.threshold = threshold
        self.predictions = predictions
        self.labels = labels
        self.errors = errors

        # ======================================================
        # PRINT
        # ======================================================
        print()
        print("=" * 60)
        print("RESULTS (CONSOLIDATED)")
        print("=" * 60)

        for k, v in results.items():
            print(f"{k}: {v}")

        # ======================================================
        # SAVE EVERYTHING
        # ======================================================
        if self.logger is not None:
            self.logger.save_all(
                model=model,
                metrics=results,  # Salva o arquivo metrics.json completo e com todas as chaves
                history=history,
                predictions=predictions,
                labels=labels,
                errors=errors,
                threshold=threshold
            )
            print("\nExperiment saved successfully.")

        self.results = results
        self.execution_time = (datetime.now() - start).total_seconds()
        print()
        print(f"Execution time: {self.execution_time:.2f} s")

        # Retorna o dicionário para o Grid Search
        return {
            "model": self.model,
            "metrics": self.results,  
            "history": self.history,
            "threshold": self.threshold,
            "execution_time": self.execution_time
        }