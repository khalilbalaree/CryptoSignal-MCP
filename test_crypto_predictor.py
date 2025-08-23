#!/usr/bin/env python3
"""
Test script for Crypto Predictor MCP Server

This script tests the core functionality of the crypto prediction service:
1. Historical data fetching
2. Model training 
3. Price prediction

Usage: python test_crypto_predictor.py
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from crypto_predictor_server import CryptoPredictionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CryptoPredictorTester:
    def __init__(self):
        self.service = CryptoPredictionService()
        self.test_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        self.test_intervals = ["1h", "4h", "1d"]
        self.results = {}
        
    async def test_get_historical_data(self):
        """Test the get_historical_data function"""
        logger.info("=== Testing Historical Data Fetching ===")
        
        for symbol in self.test_symbols:
            for interval in self.test_intervals:
                try:
                    logger.info(f"Testing {symbol} - {interval}")
                    
                    # Test with different limits
                    limits = [10, 50, 100]
                    for limit in limits:
                        start_time = time.time()
                        data = await self.service.get_historical_data(symbol, interval, limit)
                        fetch_time = time.time() - start_time
                        
                        # Validate data structure
                        if not data:
                            raise ValueError("No data returned")
                        
                        if len(data) != limit:
                            logger.warning(f"Expected {limit} records, got {len(data)}")
                        
                        # Validate first record structure
                        required_fields = ['open_time', 'open', 'high', 'low', 'close', 'volume']
                        first_record = data[0]
                        missing_fields = [field for field in required_fields if field not in first_record]
                        if missing_fields:
                            raise ValueError(f"Missing required fields: {missing_fields}")
                        
                        # Validate data types
                        if not isinstance(first_record['open'], (int, float)):
                            raise ValueError("Price data should be numeric")
                        
                        logger.info(f"✓ {symbol} {interval} limit={limit}: {len(data)} records in {fetch_time:.2f}s")
                        
                        # Store sample for analysis
                        if symbol not in self.results:
                            self.results[symbol] = {}
                        if interval not in self.results[symbol]:
                            self.results[symbol][interval] = {}
                        self.results[symbol][interval]['sample_data'] = data[:3]  # Store first 3 records
                
                except Exception as e:
                    logger.error(f"✗ Failed to fetch {symbol} {interval}: {e}")
                    
                # Rate limiting delay
                await asyncio.sleep(0.1)
        
        logger.info("Historical data testing completed\n")
    
    async def test_model_training(self):
        """Test the train_model function"""
        logger.info("=== Testing Model Training ===")
        
        # Test with 1000 historical periods as default
        test_cases = [
            ("BTCUSDT", "1h", 1000),
            ("BTCUSDT", "30m", 1000),  # Test BTC 30m accuracy
            ("BTCUSDT", "1d", 1000),   # Test BTC 1d accuracy
            ("ETHUSDT", "4h", 1000),
        ]
        
        for symbol, interval, training_periods in test_cases:
            try:
                logger.info(f"Training model for {symbol} - {interval} with {training_periods} periods")
                
                start_time = time.time()
                result = await self.service.train_model(symbol, interval, training_periods, "ensemble")
                training_time = time.time() - start_time
                
                # Validate training result
                required_keys = ['symbol', 'gradient_boosting_accuracy', 'samples', 'features', 'model_trained']
                missing_keys = [key for key in required_keys if key not in result]
                if missing_keys:
                    raise ValueError(f"Missing required keys in training result: {missing_keys}")
                
                # Validate accuracy scores
                gb_accuracy = result.get('gradient_boosting_accuracy', 0)
                if not (0 <= gb_accuracy <= 1):
                    logger.warning(f"Unusual accuracy score: {gb_accuracy}")
                
                # Check if model was stored
                if symbol not in self.service.models:
                    raise ValueError("Model was not stored in service.models")
                
                if symbol not in self.service.scalers:
                    raise ValueError("Scaler was not stored in service.scalers")
                
                logger.info(f"✓ {symbol} {interval}: Trained in {training_time:.2f}s")
                logger.info(f"  - GB Accuracy: {gb_accuracy:.4f}")
                logger.info(f"  - Features: {result.get('features', 0)}")
                logger.info(f"  - Samples: {result.get('samples', 0)}")
                logger.info(f"  - Top features: {result.get('top_features', [])[:3]}")
                
                # Store training results
                if symbol not in self.results:
                    self.results[symbol] = {}
                if interval not in self.results[symbol]:
                    self.results[symbol][interval] = {}
                self.results[symbol][interval]['training_result'] = result
                
            except Exception as e:
                logger.error(f"✗ Failed to train model for {symbol} {interval}: {e}")
        
        logger.info("Model training testing completed\n")
    
    async def test_ensemble_accuracy(self):
        """Test actual ensemble accuracy on held-out data"""
        logger.info("=== Testing Ensemble Accuracy ===")
        
        test_cases = [
            ("BTCUSDT", "1h", 1000),
            ("BTCUSDT", "30m", 1000),
            ("BTCUSDT", "1d", 1000),
            ("ETHUSDT", "4h", 1000),
        ]
        
        for symbol, interval, training_periods in test_cases:
            try:
                logger.info(f"Testing ensemble accuracy for {symbol} - {interval}")
                
                # Get full available data  
                total_data = await self.service.get_historical_data(symbol, interval, 1000)
                
                # Filter complete periods
                complete_data = self.service._filter_complete_periods(total_data)
                
                if len(complete_data) < 300:
                    logger.warning(f"Not enough data for {symbol} {interval} accuracy test")
                    continue
                
                # Split into train (80%) and test (20%)
                split_point = int(len(complete_data) * 0.8)
                train_data = complete_data[:split_point]
                test_data = complete_data[split_point:]  # Use remaining for testing
                
                if len(test_data) < 50:
                    logger.warning(f"Not enough test data for {symbol} {interval}")
                    continue
                
                # Create features for both sets
                import pandas as pd
                train_df = pd.DataFrame(train_data)
                train_df = self.service.create_features(train_df)
                
                test_df = pd.DataFrame(test_data)
                test_df = self.service.create_features(test_df)
                
                # Prepare training data
                X_train, y_train, feature_names = self.service.prepare_training_data(train_df)
                
                # Prepare test data (same features as training)
                available_features = [col for col in feature_names if col in test_df.columns]
                test_clean = test_df[available_features + ['target']].dropna()
                
                if len(test_clean) < 20:
                    logger.warning(f"Not enough clean test data for {symbol} {interval}")
                    continue
                
                X_test = test_clean[available_features].values
                y_test = test_clean['target'].values
                
                # Train ensemble (simplified version)
                import pandas as pd
                from sklearn.preprocessing import RobustScaler
                from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
                from sklearn.svm import SVC
                
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train improved models with same config as main server
                gb_model = GradientBoostingClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=4,
                    subsample=0.9, min_samples_split=10, min_samples_leaf=5,
                    random_state=42, validation_fraction=0.1, n_iter_no_change=15
                )
                svm_model = SVC(
                    kernel='rbf', C=0.5, gamma='auto', 
                    class_weight='balanced', random_state=42, probability=True
                )
                rf_model = RandomForestClassifier(
                    n_estimators=150, max_depth=8, min_samples_split=5,
                    min_samples_leaf=2, bootstrap=True, random_state=42, n_jobs=-1
                )
                
                # Create 3-model ensemble
                ensemble = VotingClassifier(
                    estimators=[('gb', gb_model), ('svm', svm_model), ('rf', rf_model)],
                    voting='soft'
                )
                
                # Train all models
                gb_model.fit(X_train_scaled, y_train)
                svm_model.fit(X_train_scaled, y_train)
                rf_model.fit(X_train_scaled, y_train)
                ensemble.fit(X_train_scaled, y_train)
                
                # Test predictions
                gb_pred = gb_model.predict(X_test_scaled)
                svm_pred = svm_model.predict(X_test_scaled)
                rf_pred = rf_model.predict(X_test_scaled)
                ensemble_pred = ensemble.predict(X_test_scaled)
                
                # Calculate accuracies
                gb_accuracy = (gb_pred == y_test).mean()
                svm_accuracy = (svm_pred == y_test).mean()
                rf_accuracy = (rf_pred == y_test).mean()
                ensemble_accuracy = (ensemble_pred == y_test).mean()
                
                logger.info(f"✓ {symbol} {interval} Accuracies:")
                logger.info(f"  - GB Test Accuracy: {gb_accuracy:.4f} ({gb_accuracy*100:.2f}%)")
                logger.info(f"  - SVM Test Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
                logger.info(f"  - RF Test Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
                logger.info(f"  - Ensemble Test Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
                logger.info(f"  - Test samples: {len(y_test)}")
                
                # Store results
                if symbol not in self.results:
                    self.results[symbol] = {}
                if interval not in self.results[symbol]:
                    self.results[symbol][interval] = {}
                    
                self.results[symbol][interval]['accuracy_test'] = {
                    'gb_test_accuracy': float(gb_accuracy),
                    'svm_test_accuracy': float(svm_accuracy),
                    'rf_test_accuracy': float(rf_accuracy),
                    'ensemble_test_accuracy': float(ensemble_accuracy),
                    'test_samples': len(y_test)
                }
                
            except Exception as e:
                logger.error(f"✗ Failed ensemble accuracy test for {symbol} {interval}: {e}")
        
        logger.info("Ensemble accuracy testing completed\n")
    
    async def test_prediction(self):
        """Test the predict_current_hour function"""
        logger.info("=== Testing Price Prediction ===")
        
        # Test prediction on symbols that should have trained models
        test_cases = [
            ("BTCUSDT", "1h", 1000),
            ("BTCUSDT", "30m", 1000),  # Test BTC 30m prediction
            ("BTCUSDT", "1d", 1000),    # Test BTC 1d prediction
            ("ETHUSDT", "4h", 1000),
        ]
        
        for symbol, interval, training_periods in test_cases:
            try:
                logger.info(f"Testing prediction for {symbol} - {interval}")
                
                start_time = time.time()
                prediction = await self.service.predict_current_hour(symbol, interval, training_periods)
                prediction_time = time.time() - start_time
                
                # Validate prediction structure
                required_keys = [
                    'symbol', 'current_price', 'prediction', 'confidence', 
                    'probability_up', 'probability_down', 'market_conditions'
                ]
                missing_keys = [key for key in required_keys if key not in prediction]
                if missing_keys:
                    raise ValueError(f"Missing required keys in prediction: {missing_keys}")
                
                # Validate prediction values
                if prediction['prediction'] not in ['UP', 'DOWN']:
                    raise ValueError(f"Invalid prediction value: {prediction['prediction']}")
                
                prob_up = prediction['probability_up']
                prob_down = prediction['probability_down']
                if not (0 <= prob_up <= 1 and 0 <= prob_down <= 1):
                    raise ValueError(f"Invalid probabilities: up={prob_up}, down={prob_down}")
                
                if abs((prob_up + prob_down) - 1.0) > 0.01:
                    logger.warning(f"Probabilities don't sum to 1: {prob_up} + {prob_down} = {prob_up + prob_down}")
                
                # Check confidence score
                confidence = prediction['confidence']
                if not (0 <= confidence <= 1):
                    raise ValueError(f"Invalid confidence score: {confidence}")
                
                logger.info(f"✓ {symbol} {interval}: Predicted in {prediction_time:.2f}s")
                logger.info(f"  - Current Price: ${prediction.get('current_price', 0):,.2f}")
                logger.info(f"  - Prediction: {prediction['prediction']}")
                logger.info(f"  - Confidence: {confidence:.4f}")
                logger.info(f"  - Prob UP: {prob_up:.4f}, DOWN: {prob_down:.4f}")
                logger.info(f"  - Market Conditions: {prediction.get('market_conditions', {})}")
                
                # Store prediction results
                if symbol not in self.results:
                    self.results[symbol] = {}
                if interval not in self.results[symbol]:
                    self.results[symbol][interval] = {}
                self.results[symbol][interval]['prediction_result'] = prediction
                
            except Exception as e:
                logger.error(f"✗ Failed to predict for {symbol} {interval}: {e}")
        
        logger.info("Price prediction testing completed\n")
    
    async def test_error_handling(self):
        """Test error handling for invalid inputs"""
        logger.info("=== Testing Error Handling ===")
        
        error_test_cases = [
            # Invalid symbol
            ("INVALIDSYMBOL", "1h", "Invalid symbol test"),
            # Invalid interval  
            ("BTCUSDT", "invalid_interval", "Invalid interval test"),
            # Very small limit
            ("BTCUSDT", "1h", "Small limit test", {"limit": 1}),
        ]
        
        for case in error_test_cases:
            if len(case) == 4:
                symbol, interval, test_name, kwargs = case
            else:
                symbol, interval, test_name = case
                kwargs = {}
            
            try:
                logger.info(f"Testing: {test_name}")
                
                if "limit" in kwargs:
                    # Test historical data with small limit
                    data = await self.service.get_historical_data(symbol, interval, kwargs["limit"])
                    logger.warning(f"Expected error for {test_name}, but got data: {len(data)} records")
                else:
                    # Test regular flow
                    data = await self.service.get_historical_data(symbol, interval, 10)
                    logger.warning(f"Expected error for {test_name}, but got data: {len(data)} records")
                
            except Exception as e:
                logger.info(f"✓ {test_name}: Correctly handled error - {type(e).__name__}: {e}")
        
        logger.info("Error handling testing completed\n")
    
    async def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        logger.info("=== Performance Benchmarks ===")
        
        # Test data fetching speed
        symbol = "BTCUSDT"
        interval = "1h"
        limits = [50, 100, 200, 500]
        
        logger.info("Data fetching performance:")
        for limit in limits:
            try:
                start_time = time.time()
                data = await self.service.get_historical_data(symbol, interval, limit)
                fetch_time = time.time() - start_time
                
                records_per_second = len(data) / fetch_time if fetch_time > 0 else 0
                logger.info(f"  - {limit} records: {fetch_time:.3f}s ({records_per_second:.1f} records/sec)")
                
            except Exception as e:
                logger.error(f"  - {limit} records: Failed - {e}")
        
        # Test training performance
        logger.info("Training performance:")
        training_sizes = [100, 200]
        for size in training_sizes:
            try:
                start_time = time.time()
                result = await self.service.train_model(symbol, interval, size)
                training_time = time.time() - start_time
                
                samples = result.get('samples', 0)
                samples_per_second = samples / training_time if training_time > 0 else 0
                logger.info(f"  - {size} periods: {training_time:.3f}s ({samples_per_second:.1f} samples/sec)")
                
            except Exception as e:
                logger.error(f"  - {size} periods: Failed - {e}")
        
        logger.info("Performance benchmarks completed\n")
    
    def generate_report(self):
        """Generate a comprehensive test report"""
        logger.info("=== Test Report ===")
        
        report = {
            "test_timestamp": datetime.now().isoformat(),
            "symbols_tested": self.test_symbols,
            "intervals_tested": self.test_intervals,
            "results_summary": {},
            "detailed_results": self.results
        }
        
        # Count successful tests
        successful_fetches = 0
        successful_trainings = 0
        successful_predictions = 0
        
        for symbol in self.results:
            for interval in self.results[symbol]:
                if 'sample_data' in self.results[symbol][interval]:
                    successful_fetches += 1
                if 'training_result' in self.results[symbol][interval]:
                    successful_trainings += 1
                if 'prediction_result' in self.results[symbol][interval]:
                    successful_predictions += 1
        
        report['results_summary'] = {
            "successful_data_fetches": successful_fetches,
            "successful_trainings": successful_trainings,
            "successful_predictions": successful_predictions,
            "total_tests": len(self.test_symbols) * len(self.test_intervals)
        }
        
        # Save report to file
        with open('/Users/zijunwu/Desktop/poly-mcp/test_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Test Report Summary:")
        logger.info(f"  - Data Fetches: {successful_fetches} successful")
        logger.info(f"  - Model Trainings: {successful_trainings} successful") 
        logger.info(f"  - Predictions: {successful_predictions} successful")
        logger.info(f"  - Report saved to: test_report.json")
    
    async def run_all_tests(self):
        """Run all tests in sequence"""
        logger.info("Starting comprehensive crypto predictor tests...\n")
        
        try:
            # Run tests in order
            await self.test_get_historical_data()
            await self.test_model_training()
            await self.test_ensemble_accuracy()
            await self.test_prediction()
            await self.test_error_handling()
            await self.test_performance_benchmarks()
            
            # Generate report
            self.generate_report()
            
            logger.info("All tests completed successfully!")
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            raise
        finally:
            # Clean up
            await self.service.client.aclose()


async def main():
    """Main test execution"""
    tester = CryptoPredictorTester()
    
    try:
        await tester.run_all_tests()
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
    finally:
        logger.info("Test execution completed.")


if __name__ == "__main__":
    asyncio.run(main())