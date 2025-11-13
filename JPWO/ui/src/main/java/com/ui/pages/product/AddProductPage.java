package com.ui.pages.product;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.Select;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

public class AddProductPage {

    private final WebDriver driver;
    private final WebDriverWait wait;

    private final By productNameField = By.cssSelector("[data-test='name']");
    private final By descriptionField = By.cssSelector("[data-test='description']");
    private final By stockField = By.cssSelector("[data-test='stock']");
    private final By priceField = By.cssSelector("[data-test='price']");
    private final By brandSelect = By.cssSelector("[data-test='brand-id']");
    private final By categorySelect = By.cssSelector("[data-test='category-id']");
    private final By imageSelect = By.cssSelector("[data-test='product-image-id']");
    private final By successMessage = By.className("alert-success");

    private final By submitButton = By.cssSelector("[data-test='product-submit']");

    public AddProductPage(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    public void open() {
        driver.get("http://localhost:4200/admin/products/add");
    }

    public void enterProductName(String name) {
        WebElement productName = wait.until(ExpectedConditions.visibilityOfElementLocated(productNameField));
        productName.clear();
        productName.sendKeys(name);
    }

    public void enterDescription(String description) {
        WebElement element = driver.findElement(descriptionField);
        element.clear();
        element.sendKeys(description);
    }

    public void enterStock(int stock) {
        WebElement element = driver.findElement(stockField);
        element.clear();
        element.sendKeys(String.valueOf(stock));
    }

    public void enterPrice(double price) {
        WebElement element = driver.findElement(priceField);
        element.clear();
        element.sendKeys(String.valueOf(price));
    }

    public void selectBrand(String brandName) {
        WebElement selectElement = driver.findElement(brandSelect);
        Select select = new Select(selectElement);
        select.selectByVisibleText(brandName);
    }

    public void selectCategory(String categoryName) {
        WebElement selectElement = driver.findElement(categorySelect);
        Select select = new Select(selectElement);
        select.selectByVisibleText(categoryName);
    }

    public void selectImage(String imageName) {
        WebElement selectElement = driver.findElement(imageSelect);
        Select select = new Select(selectElement);
        select.selectByVisibleText(imageName);
    }

    public void submitForm() throws InterruptedException {
        WebElement button = wait.until(ExpectedConditions.elementToBeClickable(submitButton));
        button.click();

        wait.until(ExpectedConditions.visibilityOfElementLocated(successMessage));
    }

    public void fillForm(String name, String description, int stock, double price,
                         String brand, String category, String image) throws InterruptedException {
        enterProductName(name);
        enterDescription(description);
        enterStock(stock);
        enterPrice(price);
        selectBrand(brand);
        selectCategory(category);
        selectImage(image);
        submitForm();
    }
}
