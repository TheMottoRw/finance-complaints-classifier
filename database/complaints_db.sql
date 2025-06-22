-- MySQL dump 10.13  Distrib 8.0.42, for Linux (x86_64)
--
-- Host: localhost    Database: complaints_db
-- ------------------------------------------------------
-- Server version	8.0.42-0ubuntu0.22.04.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `complaint`
--

DROP TABLE IF EXISTS `complaint`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `complaint` (
  `id` int NOT NULL AUTO_INCREMENT,
  `title` varchar(255) NOT NULL,
  `content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `category` varchar(255) NOT NULL,
  `department_id` int DEFAULT NULL,
  `user_id` int NOT NULL,
  `status` varchar(255) NOT NULL,
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `complaint`
--

LOCK TABLES `complaint` WRITE;
/*!40000 ALTER TABLE `complaint` DISABLE KEYS */;
INSERT INTO `complaint` VALUES (1,'Loan request','I would like to request a loan from your bank to help with my personal needs. I am interested in knowing the requirements, interest rates, and how long the process will take. Please let me know the documents I need to provide and the steps to follow. I am ready to start the application as soon as possible.','Checking or savings account',NULL,3,'pending','2025-06-22 11:55:07'),(2,'Loan request','I would like to request a loan from your bank to help with my personal needs. I am interested in knowing the requirements, interest rates, and how long the process will take. Please let me know the documents I need to provide and the steps to follow. I am ready to start the application as soon as possible.','Checking or savings account',NULL,3,'pending','2025-06-22 11:55:29'),(3,'Issues on online payment','My Visa card is about to expire soon, and I would like to know how I can renew it. I’m worried that I might not be able to use it for transactions if it expires. Please guide me on the renewal process, any fees involved, and how long it will take to receive the new card. I would like to avoid any service interruption.','Credit card or prepaid card',NULL,3,'pending','2025-06-22 11:57:12'),(4,'International transaction','I recently tried to transfer money from my account to another account within the same bank, but the transaction did not go through. The amount was deducted from my balance, but the receiver has not received the money. I kindly request your support to check what went wrong and help complete the transfer or refund the amount if necessary.','Checking or savings account',2,3,'pending','2025-06-22 12:00:34'),(5,'Vehicle loan',' would like to apply for a loan to purchase a vehicle. Please let me know the conditions for getting a car loan, including the required documents, interest rates, repayment period, and how long the approval process takes. I am ready to provide any necessary information and would appreciate your guidance on how to begin the application.','Credit reporting',5,3,'pending','2025-06-22 12:01:18');
/*!40000 ALTER TABLE `complaint` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `complaint_handling`
--

DROP TABLE IF EXISTS `complaint_handling`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `complaint_handling` (
  `id` int NOT NULL,
  `sender_id` int NOT NULL,
  `receiver_id` int NOT NULL,
  `message` text NOT NULL,
  `status` varchar(255) NOT NULL,
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `complaint_handling`
--

LOCK TABLES `complaint_handling` WRITE;
/*!40000 ALTER TABLE `complaint_handling` DISABLE KEYS */;
/*!40000 ALTER TABLE `complaint_handling` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `department`
--

DROP TABLE IF EXISTS `department`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `department` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=13 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `department`
--

LOCK TABLES `department` WRITE;
/*!40000 ALTER TABLE `department` DISABLE KEYS */;
INSERT INTO `department` VALUES (1,'Bank account or service'),(2,'Checking or savings account'),(3,'Consumer Loan'),(4,'Credit card or prepaid card'),(5,'Credit reporting'),(6,'Debt collection,credit management'),(7,'Money transfer, virtual currency, or money service'),(8,'Money transfers'),(9,'Consumer loan'),(10,'Other financial service'),(11,'Payday loan'),(12,'Administration');
/*!40000 ALTER TABLE `department` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `user`
--

DROP TABLE IF EXISTS `user`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `user` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL,
  `email` varchar(100) NOT NULL,
  `password_hash` varchar(200) NOT NULL,
  `user_type` varchar(100) DEFAULT NULL,
  `department_id` int DEFAULT NULL,
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `email` (`email`),
  KEY `department_id` (`department_id`),
  CONSTRAINT `user_ibfk_1` FOREIGN KEY (`department_id`) REFERENCES `department` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `user`
--

LOCK TABLES `user` WRITE;
/*!40000 ALTER TABLE `user` DISABLE KEYS */;
INSERT INTO `user` VALUES (1,'Admin','asua@yopmail.com','pbkdf2:sha256:260000$hZHw53cNnuT0dsf7$2bc4980d3bfe6b539176c6176a1e85fa5f2dfe7b54a84289a6ca59291f267c86','Admin',NULL,'2025-06-22 13:48:40'),(3,'Karim','karim@yopmail.com','pbkdf2:sha256:260000$n094OE1s73o8E9QJ$eec36f62134faa95da93c2d711ca8b45bdca51845ea269e12d70958f03e4ac61','customer',NULL,'2025-06-22 11:49:57');
/*!40000 ALTER TABLE `user` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-06-22 14:08:45
