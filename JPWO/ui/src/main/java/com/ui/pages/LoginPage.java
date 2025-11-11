package com.ui.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;

public class LoginPage {
    private WebDriver driver;

    // Lokatory elementów
    private By usernameField = By.id("email");
    private By passwordField = By.id("password");
    private By loginButton = By.cssSelector("[data-test='login-submit']");

    // Konstruktor
    public LoginPage(WebDriver driver) {
        this.driver = driver;
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

    public boolean isLoaded() {
        WebElement element = driver.findElement(loginButton);
        return element.isDisplayed();
    }
}
