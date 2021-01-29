--Drop All existing tables, if any
DROP TABLE IF EXISTS [ForecastTracker];

--ForecastTracker  Definition
CREATE TABLE [ForecastTracker]
(
    [ForecastID] INTEGER  PRIMARY KEY AUTOINCREMENT,
    [ForecastYear] INTEGER  NOT NULL,
    [ListingIndex] NVARCHAR(20)  NULL,
    [ListingName] NVARCHAR(100)  NOT NULL,
    [ListingTicker] NVARCHAR(20)  NOT NULL,
    [ListingType] NVARCHAR(20)  NOT NULL,
    [ListingDate] NVARCHAR(23)  NULL,
    [DataCollectionAttempt] INTEGER  DEFAULT 0  NOT NULL,
    [AvailableFrom] NVARCHAR(23)  NULL,
    [DataCollectedAt] NVARCHAR(23)  NULL,
    [RawDataFilePath] NVARCHAR(255)  NULL,
    [InputDataFilePath] NVARCHAR(255)  NULL,
    [CrossValidationCount] INTEGER  DEFAULT 0  NOT NULL,
    [AnalysisAttempt] INTEGER  DEFAULT 0  NOT NULL,
    [DataAnalyzedFrom] NVARCHAR(23)  NULL,
    [SeasonalityObserved] NVARCHAR(1000)  NULL,
    [BestModels] NVARCHAR(1000)  NULL,
    [SeasonalityPredicted] NVARCHAR(200)  NULL,
    [ConsistentModels] NVARCHAR(1000)  NULL,
    [OutputModelFolder] NVARCHAR(255)  NULL,
    [OutputDataFolder] NVARCHAR(255)  NULL,
    [OutputImageFolder] NVARCHAR(255)  NULL,
    [CreatedDate] NCHAR(23)  NOT NULL,
    [CreatedBy] NVARCHAR(50)  NOT NULL,
    [LastModifiedDate] NCHAR(23)  NULL,
    [LastModifiedBy] NVARCHAR(50)  NULL
);

CREATE UNIQUE INDEX [IPK_ForecastTracker] ON [ForecastTracker]([ForecastID]);
CREATE INDEX [IX_ForecastTracker_ForecastYear] ON [ForecastTracker] ([ForecastYear]);
CREATE INDEX [IX_ForecastTracker_ListingName] ON [ForecastTracker] ([ListingName]);
CREATE INDEX [IX_ForecastTracker_ListingTicker] ON [ForecastTracker] ([ListingTicker]);

DELETE FROM ForecastTracker WHERE ForecastID = 0;

INSERT INTO ForecastTracker
(ForecastID, ForecastYear, ListingIndex, ListingName, ListingTicker, ListingType, DataCollectionAttempt
, AvailableFrom, DataCollectedAt, RawDataFilePath, CrossValidationCount, CreatedDate, CreatedBy)
VALUES(0, 2020, 'NSE', 'Unit Testing', 'UNITTEST', 'Test', 100, '2012-06-20', '2020-01-01'
, './data/rawdata/UNITTEST.csv', -1, '2020-01-01', 'Tester');
