import torch
import torch.nn as nn


class MTLLSTNet(nn.Module):


    def __init__(

        self,

        input_size,

        hidden_size,

        output_size,

        num_layers=1,

        dropout=0.2

    ):

        super().__init__()



        self.output_size = output_size



        ####################################################
        # CNN temporal
        ####################################################


        self.conv = nn.Conv2d(

            in_channels=1,

            out_channels=32,

            kernel_size=(3,input_size)

        )


        self.relu = nn.ReLU()


        self.dropout = nn.Dropout(

            dropout

        )



        ####################################################
        # Recurrent encoder
        ####################################################


        self.gru = nn.GRU(

            input_size=32,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True

        )



        ####################################################
        # Forecast branch
        ####################################################


        self.forecast_head = nn.Linear(

            hidden_size,

            output_size

        )



        ####################################################
        # Failure classification branch
        ####################################################


        self.failure_head = nn.Sequential(

            nn.Linear(

                hidden_size,

                32

            ),


            nn.ReLU(),


            nn.Dropout(dropout),


            nn.Linear(

                32,

                1

            ),


            nn.Sigmoid()

        )



    def forward(self,x):


        ####################################################
        # x:
        # batch, window, features
        ####################################################


        x = x.unsqueeze(1)



        x = self.conv(x)



        x = self.relu(x)



        x = self.dropout(x)



        ####################################################
        # remove feature dimension
        ####################################################


        x = x.squeeze(3)



        ####################################################
        # batch, time, channels
        ####################################################


        x = x.permute(

            0,

            2,

            1

        )



        output,_ = self.gru(x)



        context = output[:,-1]



        ####################################################
        # two tasks
        ####################################################


        forecast = self.forecast_head(

            context

        )


        failure = self.failure_head(

            context

        )



        return {


            "forecast":

                forecast,


            "failure":

                failure

        }