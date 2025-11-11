package com.ui.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

public class LoginPage {

    private WebDriver driver;
    private WebDriverWait wait;

    // Lokatory elementów
    private By usernameField = By.id("email");
    private By passwordField = By.id("password");
    private By loginButton = By.cssSelector("[data-test='login-submit']");

    // Konstruktor
    public LoginPage(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    // Akcje / metody
    public void enterUsername(String username) {
        driver.findElement(usernameField).sendKeys(username);
    }

    public void enterPassword(String password) {
        driver.findElement(passwordField).sendKeys(password);
    }

    public void clickLogin() {
        driver.findElement(loginButton).click();
    }

    public AccountPage loginAs(String username, String password) {
        enterUsername(username);
        enterPassword(password);
        clickLogin();
        return new AccountPage(driver);
    }

    public void loginAsAdmin(String username, String password) {
        enterUsername(username);
        enterPassword(password);
        clickLogin();

        wait.until(ExpectedConditions.urlContains("/admin"));
    }

    public boolean isLoaded() {
        WebElement element = driver.findElement(loginButton);
        return element.isDisplayed();
    }
}
