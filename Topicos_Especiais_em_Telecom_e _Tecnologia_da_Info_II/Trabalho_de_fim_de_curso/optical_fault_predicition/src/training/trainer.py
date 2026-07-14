import copy

import torch

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler



class Trainer:


    def __init__(

        self,

        experiment,

        model,

        train_loader,

        val_loader,

        optimizer,

        criterion,

        device,

        scheduler=None

    ):


        self.experiment = experiment

        self.model = model


        self.train_loader = train_loader

        self.val_loader = val_loader


        self.optimizer = optimizer

        self.scheduler = scheduler


        self.criterion = criterion


        self.device = device


        self.epochs = experiment.epochs

        self.patience = experiment.patience



        self.best_loss = float("inf")

        self.best_state = None

        self.wait = 0



        self.clip_grad = 1.0



        self.use_amp = (

            torch.cuda.is_available()

        )


        self.scaler = GradScaler(

            enabled=self.use_amp

        )



        self.history = {


            "train_loss": [],

            "val_loss": [],


            "best_epoch": None,


        }



    ##########################################################

    # NORMALIZA SAIDA DO MODELO

    ##########################################################


    def normalize_prediction(

        self,

        prediction

    ):


        """

        Padroniza saída dos modelos.



        Aceita:



        Tensor

            modelo clássico



        tuple:

            (forecast, failure)



        dict:

            {

              forecast: tensor,

              failure: tensor

            }

        """


        if isinstance(

            prediction,

            torch.Tensor

        ):


            return {


                "forecast": prediction


            }



        if isinstance(

            prediction,

            tuple

        ):


            result = {}


            if len(prediction) >= 1:


                result["forecast"] = prediction[0]



            if len(prediction) >= 2:


                result["failure"] = prediction[1]



            return result



        if isinstance(

            prediction,

            dict

        ):


            return prediction



        raise TypeError(

            f"Unsupported model output type: {type(prediction)}"

        )
    
     ##########################################################
    # LOSS
    ##########################################################

    def compute_loss(

        self,

        prediction,

        target,

        failure_target=None

    ):


        prediction = self.normalize_prediction(

            prediction

        )


        loss = 0.0


        ######################################################

        # Forecast

        ######################################################

        if "forecast" in prediction:


            loss += self.criterion(

                prediction["forecast"],

                target

            )


        ######################################################

        # Failure (multitask)

        ######################################################

        if (

            "failure" in prediction

            and

            failure_target is not None

            and

            hasattr(

                self.experiment,

                "failure_loss"

            )

        ):


            loss += (

                self.experiment.failure_weight

                *

                self.experiment.failure_loss(

                    prediction["failure"],

                    failure_target

                )

            )


        return loss



    ##########################################################
    # TRAIN
    ##########################################################

    def train_epoch(self):


        self.model.train()


        running_loss = 0.0


        for X, y, failure in self.train_loader:


            X = X.to(

                self.device

            )


            y = y.to(

                self.device

            )


            failure = failure.to(

                self.device

            ).unsqueeze(1)


            self.optimizer.zero_grad()


            with autocast(

                enabled=self.use_amp

            ):


                prediction = self.model(

                    X

                )


                loss = self.compute_loss(

                    prediction,

                    y,

                    failure

                )


            self.scaler.scale(

                loss

            ).backward()


            torch.nn.utils.clip_grad_norm_(

                self.model.parameters(),

                self.clip_grad

            )


            self.scaler.step(

                self.optimizer

            )


            self.scaler.update()


            running_loss += loss.item()


        return (

            running_loss

            /

            len(self.train_loader)

        )



    ##########################################################
    # VALIDATION
    ##########################################################

    def validate_epoch(self):


        self.model.eval()


        running_loss = 0.0


        with torch.no_grad():


            for X, y, failure in self.val_loader:


                X = X.to(

                    self.device

                )


                y = y.to(

                    self.device

                )


                failure = failure.to(

                    self.device

                ).unsqueeze(1)


                prediction = self.model(

                    X

                )


                loss = self.compute_loss(

                    prediction,

                    y,

                    failure

                )


                running_loss += loss.item()


        return (

            running_loss

            /

            len(self.val_loader)

        )


    ##########################################################
    # BEST MODEL
    ##########################################################

    def update_best_model(

        self,

        val_loss,

        epoch

    ):


        if val_loss < self.best_loss:


            print(

                f"Validation improved "

                f"{self.best_loss:.6f}"

                f" -> "

                f"{val_loss:.6f}"

            )


            self.best_loss = val_loss


            self.best_state = copy.deepcopy(

                self.model.state_dict()

            )


            self.history["best_epoch"] = epoch + 1


            self.wait = 0



        else:


            self.wait += 1


            print(

                f"No improvement "

                f"({self.wait}/{self.patience})"

            )



        return (

            self.wait >= self.patience

        )



    ##########################################################
    # TRAINING LOOP
    ##########################################################

    def fit(self):


        print()

        print("=" * 60)

        print(

            self.experiment.name

        )

        print("=" * 60)

        print()



        for epoch in range(self.epochs):


            train_loss = self.train_epoch()


            val_loss = self.validate_epoch()



            self.history["train_loss"].append(

                train_loss

            )


            self.history["val_loss"].append(

                val_loss

            )



            print(

                f"Epoch "

                f"{epoch+1:02d}"

                f"/"

                f"{self.epochs}"

                f" | "

                f"Train "

                f"{train_loss:.6f}"

                f" | "

                f"Val "

                f"{val_loss:.6f}"

            )



            stop = self.update_best_model(

                val_loss,

                epoch

            )



            ##################################################

            # Scheduler

            ##################################################

            if self.scheduler is not None:


                if hasattr(

                    self.scheduler,

                    "step"

                ):


                    try:


                        self.scheduler.step(

                            val_loss

                        )


                    except TypeError:


                        self.scheduler.step()



            if stop:


                print()

                print(

                    "Early stopping"

                )

                break



        ######################################################
        # RESTAURA MELHOR MODELO
        ######################################################


        if self.best_state is not None:


            self.model.load_state_dict(

                self.best_state

            )



        print()


        print(

            "Best validation loss:",

            f"{self.best_loss:.6f}"

        )


        print(

            "Best epoch:",

            self.history["best_epoch"]

        )



        return self.model



    ##########################################################
    # HISTORY
    ##########################################################

    def get_history(self):


        return self.history       