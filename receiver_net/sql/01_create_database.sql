/* ==========================================================================
   01_create_database.sql
   Run ONCE as an administrator in SQL Server Management Studio (SSMS),
   connected to the server (not to a user database).

   Creates:  database CartResults, schema [cart], login/user cart_receiver.

   Replace <strong-password-from-your-vault> before running. Never commit a
   real password to source control.
   ========================================================================== */

-- 1. Database ---------------------------------------------------------------
IF DB_ID(N'CartResults') IS NULL
    CREATE DATABASE CartResults;
GO

-- Readers never block writers; matches SQLite WAL behaviour.
ALTER DATABASE CartResults SET READ_COMMITTED_SNAPSHOT ON;
GO

USE CartResults;
GO

-- 2. Schema -----------------------------------------------------------------
IF SCHEMA_ID(N'cart') IS NULL
    EXEC(N'CREATE SCHEMA cart AUTHORIZATION dbo;');
GO

-- 3. Login + user for the receiver service ----------------------------------
USE master;
GO
IF SUSER_ID(N'cart_receiver') IS NULL
    CREATE LOGIN cart_receiver WITH PASSWORD = N'<strong-password-from-your-vault>';
GO

USE CartResults;
GO
IF USER_ID(N'cart_receiver') IS NULL
    CREATE USER cart_receiver FOR LOGIN cart_receiver;
GO

ALTER ROLE db_datareader ADD MEMBER cart_receiver;
ALTER ROLE db_datawriter ADD MEMBER cart_receiver;
GRANT EXECUTE ON SCHEMA::cart TO cart_receiver;
GO

/* The service account above needs db_owner only for the initial deploy of
   02_create_tables.sql. At runtime datareader + datawriter is enough.        */
