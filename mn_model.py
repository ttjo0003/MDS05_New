# ======================
# MODEL
# ======================

class MobileNetLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=1):
        super().__init__()

        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.cnn = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_dim = 1280

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, num_classes)
        )

    def forward(self, x):
        # x: B, T, C, H, W
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)

        feat = self.cnn(x)
        feat = self.pool(feat)
        feat = feat.view(B, T, self.feature_dim)

        lstm_out, _ = self.lstm(feat)

        final_feat = lstm_out[:, -1, :]

        logits = self.classifier(final_feat)
        return logits
