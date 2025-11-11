package com.ui.tests;

import com.ui.pages.AccountPage;
import com.ui.pages.LoginPage;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class LoginTest {
    private WebDriver driver;
    private LoginPage loginPage;

    @BeforeEach
    void setUp() {
        System.setProperty("webdriver.chrome.driver", "C:\\Users\\Janek\\Downloads\\chromedriver-win64\\chromedriver.exe");
        driver = new ChromeDriver();
        driver.manage().window().maximize();

        driver.get("http://localhost:4200/auth/login");
        loginPage = new LoginPage(driver);
    }

    @Test
    void shouldLoginSuccessfully() {
        assertTrue(loginPage.isLoaded(), "Login page did not load properly");

        AccountPage accountPage = loginPage.loginAs("customer@practicesoftwaretesting.com", "welcome01");
        assertTrue(accountPage.isLoaded(), "Account page did not load properly");
        assertTrue(accountPage.getTitle().contains("My account"),
                "User was not redirected to the dashboard");
    }

    @AfterEach
    void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
}
