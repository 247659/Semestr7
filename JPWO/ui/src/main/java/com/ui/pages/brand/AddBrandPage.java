package com.ui.pages.brand;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

public class AddBrandPage {

    private final WebDriver driver;
    private final WebDriverWait wait;

    private final By brandNameField = By.cssSelector("[data-test='name']");
    private final By slugField = By.cssSelector("[data-test='slug']");
    private final By successMessage = By.className("alert-success");

    private final By submitButton = By.cssSelector("[data-test='brand-submit']");

    public AddBrandPage(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    public void open() {
        driver.get("http://localhost:4200/admin/brands/add");
    }

    public void enterBrandName(String name) {
        WebElement productName = wait.until(ExpectedConditions.visibilityOfElementLocated(brandNameField));
        productName.clear();
        productName.sendKeys(name);
    }

    public void enterSlug(String slug) {
        WebElement element = driver.findElement(slugField);
        element.clear();
        element.sendKeys(slug);
    }


    public void submitForm() {
        WebElement button = wait.until(ExpectedConditions.elementToBeClickable(submitButton));
        button.click();

        wait.until(ExpectedConditions.visibilityOfElementLocated(successMessage));
    }

    public void fillForm(String name, String slug) {
        enterBrandName(name);
        enterSlug(slug);
        submitForm();
    }
}
